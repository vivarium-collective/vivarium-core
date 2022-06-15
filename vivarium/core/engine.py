"""
==========
Engine
==========

Engine runs the simulation.
"""

import cProfile
import pstats
import os
import logging as log
import pprint
import re
from typing import (
    Any, Dict, Optional, Union, Tuple, Callable, Iterable, List, Set,
    cast, Sequence)
import math
import datetime
import time as clock
import uuid

import networkx as nx
import pytest

from vivarium.core.store import (
    hierarchy_depth, Store, generate_state, view_values)
from vivarium.core.emitter import get_emitter, Emitter
from vivarium.core.process import (
    Process,
    ParallelProcess,
    Step,
)
from vivarium.core.composer import Composite
from vivarium.core.serialize import serialize_value
from vivarium.library.topology import (
    get_in,
    delete_in,
    assoc_path,
    inverse_topology,
    normalize_path,
)
from vivarium.library.dict_utils import apply_func_to_leaves
from vivarium.core.types import (
    HierarchyPath, Topology, State, Update, Processes, Steps,
    Flow, Schema)

pretty = pprint.PrettyPrinter(indent=2)


def pp(x: Any) -> None:
    """Print ``x`` in a pretty format."""
    pretty.pprint(x)


def pf(x: Any) -> str:
    """Format ``x`` for display."""
    return pretty.pformat(x)


log.basicConfig(level=os.environ.get("LOGLEVEL", log.WARNING))


def starts_with(
        a_list: HierarchyPath,
        sub: HierarchyPath,
) -> bool:
    """Check whether one path is a prefix of another.

    Args:
        a_list: Path to look for prefix in.
        sub: Prefix.

    Returns:
        True if ``sub`` is a prefix of ``a_list``; False otherwise.
    """
    return len(sub) <= len(a_list) and all(
        a_list[i] == el
        for i, el in enumerate(sub))


def invert_topology(
        update: Update,
        args: Tuple[HierarchyPath, Topology],
) -> State:
    """Wrapper function around ``inverse_topology``.

    Wraps :py:func:`vivarium.library.topology.inverse_topology`.

    Updates are produced relative to the process that produced them. To
    transform them such that they are relative to the root of the
    simulation hierarchy, this function "inverts" a topology.

    Args:
        update: The update.
        args: Tuple of the path to which the update is relative and the
            topology.

    Returns:
        The update, relative to the root of ``path``.
    """
    path, topology = args
    return inverse_topology(path[:-1], update, topology)


def timestamp(dt: Optional[Any] = None) -> str:
    """Get a timestamp of the form ``YYYYMMDD.HHMMSS``.

    Args:
        dt: Datetime object to generate timestamp from. If not
            specified, the current time will be used.

    Returns:
        Timestamp.
    """
    if not dt:
        dt = datetime.datetime.now()
    return "%04d%02d%02d.%02d%02d%02d" % (
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second)


def empty_front(t: float) -> Dict[str, Union[float, dict]]:
    return {
        'time': t,
        'update': {}}


class Defer:
    def __init__(
            self,
            defer: Any,
            f: Callable,
            args: Tuple,
    ) -> None:
        """Allows for delayed application of a function to an update.

        The object simply holds the provided arguments until it's time
        for the computation to be performed. Then, the function is
        called.

        Args:
            defer: An object with a ``.get_command_result()`` method
                whose output will be passed to the function. For
                example, the object could be an
                :py:class:`vivarium.core.process.Process` object whose
                ``.get_command_result()`` method will return the process
                update.
            function: The function. For example,
                :py:func:`invert_topology` to transform the returned
                update.
            args: Passed as the second argument to the function.
        """
        self.defer = defer
        self.f = f
        self.args = args

    def get(self) -> Update:
        """Perform the deferred computation.

        Returns:
            The result of calling the function.
        """
        return self.f(
            self.defer.get_command_result(),
            self.args)


class EmptyDefer(Defer):
    def __init__(self) -> None:
        function = lambda update, arg: update
        args = ()
        super().__init__(None, function, args)

    def get(self) -> Update:
        return {}


class _StepGraph:
    """A dependency graph of :term:`steps`.

    A step is just a Process object that has dependencies on other
    steps. Unlike processes, which can be run in any order, steps run
    every timestep. In a given timestep, each step must not run until
    all its dependency steps have run and had their updates applied.

    Note that the constructor uses any provided arguments without
    copying them.

    Attributes:
        graph: A NetworkX DiGraph with an edge for each dependency
            relationship and a node for each step path. If the step at
            path ``a`` depends on the step at path ``b``, then the graph
            will contain an edge from ``b`` to ``a``. This means that a
            topological sort of the graph produces a valid runtime order
            for the step. Note that the graph must be a DAG.
        sequential_steps: A list of paths for steps that should run
            sequentially and before the steps in ``graph``. This is
            where we store legacy :term:`derivers` for
            backwards-compatibility.
    """

    def __init__(
            self,
            graph: Optional[nx.DiGraph] = None,
            sequential_steps: Optional[List[HierarchyPath]] = None
            ) -> None:
        self._graph = graph or nx.DiGraph()
        self._sequential_steps: List[HierarchyPath] = (
            sequential_steps or [])

    def _validate(self) -> None:
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError('Step graph must be a DAG.')
        graph_steps = set(self._graph.nodes)
        sequential_steps = set(self._sequential_steps)
        intersection = graph_steps & sequential_steps
        if intersection:
            raise ValueError(
                'self._graph and self._sequential_steps have '
                f'overlapping steps: {intersection}')

    def add(
            self,
            path: HierarchyPath,
            dependencies: Iterable[HierarchyPath]) -> None:
        """Add a step to the graph.

        Args:
            path: The step object's path in the hierarchy.
            dependencies: The hierarchy paths to each dependency.

        Raises:
            ValueError: If the graph produced by adding the step is not
                a DAG.
        """
        self._graph.add_node(path)
        for dependency in dependencies:
            self._graph.add_edge(dependency, path)
        self._validate()

    def add_sequential(
            self,
            path: HierarchyPath) -> None:
        """Add a step that is meant to run sequentially.

        Legacy steps (:term:`derivers`) were meant to run sequentially
        instead of being provided as a dependency graph. To support
        these legacy steps, this method adds a step that will run after
        all previously-added sequential steps and before any
        non-sequential steps.

        Args:
            path: The path to the step in the hierarchy.
        """
        self._sequential_steps.append(path)
        self._validate()

    def get_execution_layers(self) -> List[Set[HierarchyPath]]:
        """Get step execution layers, with steps represnted by paths.

        An execution layer is a set of steps that can be executed in
        parallel. The graph's execution layers are an ordered list of
        these layers such that:

        * For a given layer, every step in the layer may be executed as
          soon as all steps in preceding layers have been executed.
        * Every step is in as early a layer as possible.

        In other words, the execution layers are the topological
        generations of the graph, prepended by any sequential steps,
        each in its own layer and in the order in which they were added.

        Returns:
            An ordered list of the execution layers, with each step
            represented by its path.
        """
        layers = nx.topological_generations(self._graph)
        to_return = [set([step]) for step in self._sequential_steps]
        to_return += [set(layer) for layer in layers]
        return to_return

    def remove(self, path: HierarchyPath) -> None:
        """Delete a step based on its path.

        Args:
            path: Hierarhcy path of the step to delete.
        """
        if path in self._sequential_steps:
            self._sequential_steps.remove(path)
            return
        to_delete = nx.algorithms.dag.descendants(self._graph, path)
        to_delete.add(path)
        for path_to_delete in to_delete:
            self._graph.remove_node(path_to_delete)

    def copy(self) -> '_StepGraph':
        """Create a copy of self.

        Returns:
            A new _StepGraph with a copy of self's graph.
        """
        new = self.__class__(
            self._graph.copy(), self._sequential_steps.copy())
        return new


class Engine:
    def __init__(
            self,
            composite: Optional[Composite] = None,
            processes: Optional[Processes] = None,
            steps: Optional[Steps] = None,
            flow: Optional[Flow] = None,
            topology: Optional[Topology] = None,
            store: Optional[Store] = None,
            initial_state: Optional[State] = None,
            experiment_id: Optional[str] = None,
            experiment_name: Optional[str] = None,
            metadata: Optional[dict] = None,
            description: str = '',
            emitter: Union[str, dict] = 'timeseries',
            store_schema: Optional[dict] = None,
            emit_topology: bool = True,
            emit_processes: bool = False,
            emit_config: bool = False,
            emit_step: float = 1,
            display_info: bool = True,
            progress_bar: bool = False,
            global_time_precision: Optional[int] = None,
            profile: bool = False,
    ) -> None:
        """Defines simulations

        Args:
            composite: A :term:`Composite`, which specifies the processes,
                steps, flow, and topology. This is an alternative to passing
                in processes and topology dict, which can not be loaded
                at the same time.
            processes: A dictionary that maps :term:`process` names to
                process objects. You will usually get this from the
                ``processes`` key of the dictionary from
                :py:meth:`vivarium.core.composer.Composer.generate`.
            steps: A dictionary that maps :term:`step` names to step
                objects. You will usually get this from the ``steps``
                key of the dictionary from
                :py:meth:`vivarium.core.composer.Composer.generate`.
            flow: A dictionary that maps :term:`step` names to sequences
                of paths to the steps that the step depends on. You will
                usually get this from the ``flow`` key of the dictionary
                from
                :py:meth:`vivarium.core.composer.Composer.generate`.
            topology: A dictionary that maps process names to
                sub-dictionaries. These sub-dictionaries map the
                process's port names to tuples that specify a path
                through the :term:`tree` from the :term:`compartment`
                root to the :term:`store` that will be passed to the
                process for that port.
            store: A pre-loaded Store. This is an alternative to passing
                in processes and topology dict, which can not be loaded
                at the same time. Note that if you provide this
                argument, you must ensure that all parallel processes
                (i.e. :py:class:`vivarium.core.process.Process` objects
                with the ``parallel`` attribute set to ``True``) are
                instances of
                :py:class:`vivarium.core.process.ParallelProcess`. This
                constructor converts parallel processes to
                ``ParallelProcess`` objects automatically if you do not
                provide this ``store`` argument.
            initial_state: By default an empty dictionary, this is the
                initial state of the simulation.
            experiment_id: A unique identifier for the experiment. A
                UUID will be generated if none is provided.
            metadata: A dictionary with additional data about the experiment,
                which is saved by the emitter with the configuration.
            description: A description of the experiment. A blank string
                by default.
            emitter: An emitter configuration which must conform to the
                specification in the documentation for
                :py:func:`vivarium.core.emitter.get_emitter`. The
                experiment ID will be added to the dictionary you
                provide as the value for the key ``experiment_id``.
            display_info: prints experiment info
            progress_bar: shows a progress bar
            global_time_precision: an optional int that sets the decimal
                precision of global_time. This is useful for remove floating-
                point rounding errors for the time keys of saved states.
            store_schema: An optional dictionary to expand the store hierarchy
                configuration, and also to turn emits on or off. The dictionary
                needs to be structured as a hierarchy, which will expand the
                existing store hierarchy. Setting an emit value for a branch
                node will set the emits of all the leaves to that value.
            emit_topology: If True, this will emit the topology with the
                configuration data.
            emit_processes: If True, this will emit the serialized
                processes with the configuration data.
            emit_config: If True, this will emit the serialized initial
                state with the configuration data.
            profile: Whether to profile the simulation with cProfile.
        """
        self.profiler: Optional[cProfile.Profile] = None
        if profile:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        self.stats_objs: List[pstats.Stats] = []
        self.stats: Optional[pstats.Stats] = None

        self.experiment_id = experiment_id or str(uuid.uuid1())
        self.initial_state = initial_state or {}
        self.emit_step = emit_step

        # make the processes, topology, steps, flow, and store
        self._make_store(store, composite, processes, steps, flow, topology)

        # display settings
        self.experiment_name = experiment_name or self.experiment_id
        self.metadata = metadata
        self.description = description
        self.display_info = display_info
        self.global_time_precision = global_time_precision
        self.progress_bar = progress_bar
        self.time_created = timestamp()
        if self.display_info:
            self._print_display()

        # get a mapping of all paths to processes
        self.process_paths: Dict[HierarchyPath, Process] = {}
        self._step_graph = _StepGraph()
        self._step_paths: Dict[HierarchyPath, Process] = {}
        self._find_process_paths(self.processes, self.flow)
        self._find_step_paths(self.steps, self.flow)
        self._validate_steps_and_flow(self._step_paths, self.flow)

        # emitter settings
        emitter_config = emitter
        if isinstance(emitter_config, str):
            emitter_config = {'type': emitter_config}
        else:
            emitter_config = dict(emitter_config)
        emitter_config['experiment_id'] = self.experiment_id
        self.emitter: Emitter = get_emitter(emitter_config)

        # override emit settings in store
        if store_schema:
            self.state._apply_config(store_schema)

        # settings for self._emit_configuration()
        self.emit_topology = emit_topology
        self.emit_processes = emit_processes
        self.emit_config = emit_config

        # initialize global time
        self.global_time = 0.0

        # front tracks how far each process has been simulated in time,
        # and also holds the processes' updates before they are applied.
        self.front: Dict = {
            path: empty_front(self.global_time)
            for path in self.process_paths}

        # run the steps
        self.run_steps()

        # run the emitter
        self._emit_configuration()
        self._emit_store_data()

        # logging information
        log.info('experiment %s', str(self.experiment_id))

        log.info('\nPROCESSES:')
        log.info(pf(self.processes))

        log.info('\nTOPOLOGY:')
        log.info(pf(self.topology))

    @staticmethod
    def _validate_steps_and_flow(
            step_paths: Dict[HierarchyPath, Process],
            flow: Flow,
            path: HierarchyPath = tuple()) -> None:
        '''Check that every Step in flow is in steps.'''
        for key, sub_flow in flow.items():
            if isinstance(sub_flow, dict):
                Engine._validate_steps_and_flow(
                    step_paths, sub_flow, path + (key,))
            else:
                assert isinstance(sub_flow, list)
                for dependency in sub_flow:
                    dependency = path + dependency
                    if dependency not in step_paths:
                        raise Exception(
                            f'Unknown dependency step {dependency} is '
                            'in the flow')

    def _make_store(
            self,
            store: Store = None,
            composite: Composite = None,
            processes: Processes = None,
            steps: Steps = None,
            flow: Flow = None,
            topology: Topology = None,
    ) -> None:
        """
        If a :term:`Store` is provided, retrieve the :term:`Processes`,
        :term:`Steps`, :term:`Flow`, and :term:`Topology`. If a
        :term:`Composite` or its attributes are provided, create a
        store. These are interchangeable.
        """
        if not store:
            if (processes and topology) or (steps and topology):
                self.processes = processes or {}
                self.steps = steps or {}
                self.flow = flow or {}
                self.topology = topology
            elif composite:
                self.processes = composite['processes']
                self.steps = composite['steps']
                self.flow = composite['flow']
                self.topology = composite['topology']
                self.initial_state = composite['state'] or self.initial_state
            else:
                raise Exception(
                    'load either composite, store, or '
                    '(processes and topology) into Engine')

            self.processes = cast(
                Dict[str, Any],
                self._parallelize_processes(self.processes)
            )
            self.steps = cast(
                Dict[str, Any],
                self._parallelize_processes(self.steps)
            )

            # initialize the store
            self.state: Store = generate_state(
                self.processes,
                self.topology,
                self.initial_state,
                self.steps,
                self.flow,
            )

        else:
            self.state = store
            self.state.set_value(self.initial_state)
            # build the processes' views
            self.state.build_topology_views()
            # get processes and topology from the store
            self.processes = self.state.get_processes()
            self.steps = self.state.get_steps() or {}
            self.flow = self.state.get_flow() or {}
            self.topology = self.state.get_topology()

    def _parallelize_processes(
            self, processes: Any) -> Union[dict, Process]:
        '''Replace parallel processes with ParallelProcess objects.'''
        if isinstance(processes, Process):
            if processes.parallel and not isinstance(
                    processes, ParallelProcess):
                processes = ParallelProcess(
                    processes, bool(self.profiler), self.stats_objs)
        elif isinstance(processes, dict):
            processes = {
                key: self._parallelize_processes(value)
                for key, value in processes.items()
            }
        else:
            raise AssertionError(f'Unrecognized collection: {processes}')
        return processes

    def _add_step_path(
            self,
            step: Step,
            path: HierarchyPath,
            # None if deriver, empty list if no dependencies.
            relative_dependencies: Optional[Sequence[HierarchyPath]],
    ) -> None:
        assert step.is_step()
        self._step_paths[path] = step
        if relative_dependencies is None:
            self._step_graph.add_sequential(path)
            return
        dependencies = [
            path + ('..',) + dep for dep in relative_dependencies]
        norm_dependencies = [
            normalize_path(dep) for dep in dependencies]
        self._step_graph.add(path, norm_dependencies)

    def _add_process_path(
            self,
            process: Process,
            path: HierarchyPath,
            flow: Flow,
    ) -> None:
        if process.is_step():
            # warnings.warn(
            #     f'Found a step {path} in the processes dict. This is '
            #     'deprecated. Steps should be specified in the steps '
            #     'dict instead.',
            #     category=FutureWarning,
            # )
            step = cast(Step, process)
            self._add_step_path(step, path, get_in(flow, path))
        else:
            self.process_paths[path] = process

    def _find_process_paths(
            self,
            processes: Processes,
            flow: Flow,
    ) -> None:
        tree = hierarchy_depth(processes)
        for path, process in tree.items():
            self._add_process_path(process, path, flow)

    def _find_step_paths(
            self,
            steps: Steps,
            flow: Flow,
    ) -> None:
        tree = hierarchy_depth(steps)
        for path, step in tree.items():
            self._add_step_path(step, path, get_in(flow, path))

    def _emit_configuration(self) -> None:
        """Emit experiment configuration."""
        data: Dict[str, Any] = {
            'time_created': self.time_created,
            'experiment_id': self.experiment_id,
            'name': self.experiment_name,
            'description': self.description,
            'metadata': self.metadata,
            'topology': self.topology
            if self.emit_topology else None,
            'processes': self.processes
            if self.emit_processes else None,
            'state': self.state.get_config()
            if self.emit_config else None,
        }
        emit_config: Dict[str, Any] = {
            'table': 'configuration',
            'data': serialize_value(data)
        }
        self.emitter.emit(emit_config)

    def _emit_store_data(self) -> None:
        """Emit the current simulation state.
        Only variables with ``_emit=True`` are emitted.
        """
        data = self.state.emit_data()
        data.update({
            'time': self.global_time})
        emit_config = {
            'table': 'history',
            'data': serialize_value(data)}
        self.emitter.emit(emit_config)

    def _invoke_process(
            self,
            process: Process,
            interval: float,
            states: State,
    ) -> Any:
        """Trigger computation of a process's update.

        To allow processes to run in parallel, this function only
        triggers update computation. When the function exits,
        computation may not be complete.

        Args:
            process: The process.
            interval: The timestep for which to compute the update.
            states: The simulation state to pass to
                :py:meth:`vivarium.core.process.Process.next_update`.

        Returns:
            The deferred simulation update, for example a
            :py:class:`vivarium.core.process.ParallelProcess`.
        """
        process.send_command('next_update', (interval, states))
        return process

    def _process_update(
            self,
            path: HierarchyPath,
            process: Process,
            store: Store,
            states: State,
            interval: float,
    ) -> Tuple[Defer, Store]:
        """Start generating a process's update.

        This function is similar to :py:meth:`_invoke_process` except in
        addition to triggering the computation of the process's update
        (by calling ``_invoke_process``), it also generates a
        :py:class:`Defer` object to transform the update into absolute
        terms.

        Args:
            path: Path to process.
            process: The process.
            store: The store at ``path``.
            states: Simulation state to pass to process's
                ``next_update`` method.
            interval: Timestep for which to compute the update.

        Returns:
            Tuple of the deferred update (in absolute terms) and
            ``store``.
        """

        update = self._invoke_process(
            process,
            interval,
            states)

        absolute = Defer(
            update,
            invert_topology,
            (path, store.topology))

        return absolute, store

    def _process_state(
            self,
            path: HierarchyPath,
    ) -> Tuple[Store, State]:
        """Get the simulation state for a process's ``next_update``.

        Before computing an update, we have to collect the simulation
        variables the processes expects.

        Args:
            path: Path to the process.

        Returns:
            Tuple of the store at ``path`` and a collection of state
            variables in the form the process expects.
        """
        store = self.state.get_path(path)
        assert isinstance(store.value, Process)

        # translate the values from the tree structure into the form
        # that this process expects, based on its declared topology
        topology_view = store.topology_view
        assert topology_view is not None, \
            f"store at path {path} does not have a topology_view"
        states = view_values(topology_view)

        return store, states

    def _calculate_update(
            self,
            path: HierarchyPath,
            process: Process,
            interval: float
    ) -> Tuple[Defer, Store]:
        """Calculate a process's update.

        Args:
            path: Path to process.
            process: The process.
            interval: Timestep to compute update for.

        Returns:
            Tuple of the deferred update (relative to the root of
            ``path``) and the store at ``path``.
        """
        store, states = self._process_state(path)
        if process.update_condition(interval, states):
            return self._process_update(
                path, process, store, states, interval)
        return (EmptyDefer(), store)

    def apply_update(
            self,
            update: Update,
            state: Store
    ) -> bool:
        """Apply an update to the simulation state.

        Args:
            update: The update to apply. Must be relative to ``state``.
            state: The store to which the update is relative (usually
                root of simulation state. We need this so to preserve
                the "perspective" from which the update was generated.

        Return:
            a bool indicating whether the topology_views expired.
        """

        if not update:
            return False

        (
            topology_updates, process_updates, step_updates,
            flow_updates, deletions, view_expire
        ) = self.state.apply_update(update, state)

        process_updates = [
            (path, self._parallelize_processes(process))
            for path, process in process_updates
        ]
        step_updates = [
            (path, self._parallelize_processes(step))
            for path, step in step_updates
        ]
        # Make sure the Store contains the parallelized processes.
        for path, process in process_updates:
            self.state.get_path(path).value = process
        for path, step in step_updates:
            self.state.get_path(path).value = step

        flow_update_dict = dict(flow_updates)

        if topology_updates:
            for path, topology_update in topology_updates:
                assoc_path(self.topology, path, topology_update)

        if process_updates:
            for path, process in process_updates:
                assoc_path(self.processes, path, process)
                self._add_process_path(process, path, {})

        if step_updates:
            for path, step in step_updates:
                dependencies = flow_update_dict.get(path)
                assoc_path(self.steps, path, step)
                self._add_step_path(step, path, dependencies)

        if deletions:
            for deletion in deletions:
                self._delete_path(deletion)

        return view_expire

    def _delete_path(
            self,
            deletion: HierarchyPath
    ) -> None:
        """Delete a store from the state.

        Args:
            deletion: Path to store to delete.
        """
        delete_in(self.processes, deletion)
        delete_in(self.steps, deletion)
        delete_in(self.topology, deletion)

        for path in list(self.process_paths.keys()):
            if starts_with(path, deletion):
                del self.process_paths[path]

        for path in list(self._step_paths):
            if starts_with(path, deletion):
                try:
                    self._step_graph.remove(path)
                except nx.exception.NetworkXError as e:
                    # The step might have been deleted already.
                    msg = f'The node {path} is not in the graph.'
                    if e.args[0] != msg:
                        raise e
                del self._step_paths[path]

    def run_steps(self) -> None:
        """Run all the steps in the simulation."""
        layers = self._step_graph.get_execution_layers()
        for layer in layers:
            deferred_updates: List[Tuple[Defer, Store]] = []
            for path in layer:
                step = self._step_paths.get(path)
                if not step:
                    # Step was deleted by a previous step.
                    continue
                # Timestep shouldn't influence steps.
                # TODO(jerry): Do something cleaner than having
                #  generate_paths() add a schema attribute to the Deriver.
                #  PyCharm's type check reports:
                #    Type Process doesn't have expected attribute 'schema'
                update, store = self._calculate_update(
                    path, step, 0)
                deferred_updates.append((update, store))

            view_expire = False
            for update, store in deferred_updates:
                view_expire_update = self.apply_update(update.get(), store)
                view_expire = view_expire or view_expire_update

            if view_expire:
                self.state.build_topology_views()

    def _send_updates(
            self,
            update_tuples: list
    ) -> None:
        """Apply updates and run steps.

        Args:
            update_tuples: List of tuples ``(update, state)`` where
                ``state`` is the store from whose perspective the update
                was generated.
        """
        view_expire = False
        for update_tuple in update_tuples:
            update, state = update_tuple
            view_expire_update = self.apply_update(update.get(), state)
            view_expire = view_expire or view_expire_update

        if view_expire:
            self.state.build_topology_views()

        self.run_steps()

    def update(
            self,
            interval: float,
    ) -> None:
        """
        Run each process for the given interval and force them to complete
        at the end of the interval. See ``run_for`` for the keyword args.
        """
        clock_start = clock.time()
        self.run_for(
            interval=interval,
            force_complete=True,
        )
        self._check_complete()
        runtime = clock.time() - clock_start
        if self.display_info:
            self._print_summary(runtime)

    def complete(self) -> None:
        """Force all processes to complete at the current global time"""
        self.run_for(interval=0, force_complete=True)
        self._check_complete()

    def _check_complete(self) -> None:
        """Check that all processes completed"""
        for path, advance in self.front.items():
            assert advance['time'] == self.global_time, \
                f"the process at path {path} is at time {advance['time']} " \
                f"instead of completing at global time {self.global_time}"
            assert len(advance['update']) == 0, \
                f"the process at path {path} is an unapplied update"

    def _remove_deleted_processes(self) -> None:
        '''Remove deleted processes from the front.'''
        self.front = {
            path: progress
            for path, progress in self.front.items()
            if path in self.process_paths}

    def run_for(
            self,
            interval: float,
            force_complete: bool = False,
    ) -> None:
        """Run each process within the given interval and update their states.

        Args:
            interval: the amount of time to simulate the composite.
            force_complete: a bool indicating whether to force processes
                to complete at the end of the interval.
        """
        end_time = self.global_time + interval
        emit_time = self.global_time + self.emit_step

        while self.global_time < end_time or force_complete:
            full_step = math.inf
            self._remove_deleted_processes()

            # processes at quiet paths don't meet their execution condition,
            # but still advance in time
            quiet_paths = []

            # go through each process and find those that are able to update
            # based on their most recent update time being less than global time
            for path, process in self.process_paths.items():
                if path not in self.front:
                    self.front[path] = empty_front(self.global_time)
                process_time = self.front[path]['time']

                if process_time <= self.global_time:

                    # get the time step
                    store, states = self._process_state(path)
                    process_timestep = process.calculate_timestep(states)

                    if force_complete:
                        # force the process to complete at end_time
                        future = min(process_time + process_timestep, end_time)
                    else:
                        future = process_time + process_timestep
                    if self.global_time_precision is not None:
                        # set future time based on global_time_precision
                        future = round(future, self.global_time_precision)

                    if future <= end_time:

                        # calculate the update for this process
                        if process.update_condition(process_timestep, states):
                            update = self._process_update(
                                path, process, store, states, process_timestep)

                            # update front, to be applied at its projected time
                            self.front[path]['time'] = future
                            self.front[path]['update'] = update

                            # absolute timestep
                            timestep = future - self.global_time
                            if timestep < full_step:
                                full_step = timestep
                        else:
                            # mark this path "quiet" so its time can be advanced
                            self.front[path]['update'] = (EmptyDefer(), store)
                            quiet_paths.append(path)
                    else:
                        # absolute timestep
                        timestep = future - self.global_time
                        if timestep < full_step:
                            full_step = timestep

                else:
                    # don't shoot past processes that didn't run this time
                    process_delay = process_time - self.global_time
                    if process_delay < full_step:
                        full_step = process_delay

            # apply updates based on process times in self.front
            if full_step == math.inf:
                # no processes ran, jump to next process
                next_event = end_time
                for path in self.front.keys():
                    if self.front[path]['time'] < next_event:
                        next_event = self.front[path]['time']
                self.global_time = next_event

            elif self.global_time + full_step <= end_time:
                # at least one process ran within the interval
                # increase the time, apply updates, and continue
                self.global_time += full_step

                # advance all quiet processes to current time
                for quiet in quiet_paths:
                    self.front[quiet]['time'] = self.global_time

                # apply updates that are behind global time
                updates = []
                paths = []
                for path, advance in self.front.items():
                    if advance['time'] <= self.global_time \
                            and advance['update']:
                        new_update = advance['update']
                        updates.append(new_update)
                        advance['update'] = {}
                        paths.append(path)

                self._send_updates(updates)

                # display and emit
                if self.progress_bar:
                    print_progress_bar(self.global_time, end_time)
                if self.emit_step == 1:
                    self._emit_store_data()
                elif emit_time <= self.global_time:
                    while emit_time <= self.global_time:
                        self._emit_store_data()
                        emit_time += self.emit_step

            else:
                # all processes have run past the interval
                self.global_time = end_time

            if force_complete and self.global_time == end_time:
                force_complete = False

    @staticmethod
    def _end_process_if_parallel(process: Process) -> None:
        if process.parallel:
            assert isinstance(process, ParallelProcess)
            process.end()

    def end(self) -> None:
        """Terminate all processes running in parallel.

        This MUST be called at the end of any simulation with parallel
        processes. This function also ends profiling and computes
        profiling stats, including stats from parallel sub-processes.
        These stats are stored in ``self.stats``.
        """
        apply_func_to_leaves(
            self.processes, self._end_process_if_parallel)
        if self.profiler:
            self.profiler.disable()
            total_stats = pstats.Stats(self.profiler)
            for stats in self.stats_objs:
                total_stats.add(stats)
            self.stats = total_stats

    def _print_display(self) -> None:
        """Print simulation metadata."""
        date, time = self.time_created.split('.')
        print('\nSimulation ID: {}'.format(self.experiment_id))
        print('Created: {} at {}'.format(
            date[4:6] + '/' + date[6:8] + '/' + date[0:4],
            time[0:2] + ':' + time[2:4] + ':' + time[4:6]))
        if self.experiment_name is not self.experiment_id:
            print('Name: {}'.format(self.experiment_name))
        if self.description:
            print('Description: {}'.format(self.description))

    def _print_summary(
            self,
            runtime: float
    ) -> None:
        """Print summary of simulation runtime."""
        if runtime < 1:
            print('Completed in {:.6f} seconds'.format(runtime))
        else:
            print('Completed in {:.2f} seconds'.format(runtime))


def print_progress_bar(
        iteration: float,
        total: float,
        decimals: float = 1,
        length: int = 50,
) -> None:
    """Create terminal progress bar

    Args:
        iteration: Current iteration
        total: Total iterations
        decimals: Positive number of decimals in percent complete
        length: Character length of bar
    """
    progress: str = ("{0:." + str(decimals) + "f}").format(total - iteration)
    filled_length: int = int(length * iteration // total)
    filled_bar = '█' * filled_length + '-' * (length - filled_length)
    print(
        f'\rProgress:|{filled_bar}| {progress}/{float(total)} '
        f'simulated seconds remaining    ', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def test_flow_with_extra_step() -> None:
    class StepA(Step):

        def ports_schema(self) -> Schema:
            return {
                'a': {'_default': 1}
            }

        def next_update(self, timestep: float, states: State) -> Update:
            return {}

    class ProcessB(Process):

        def ports_schema(self) -> Schema:
            return {
                'b': {'_default': 1}
            }

        def next_update(self, timestep: float, states: State) -> Update:
            return {}


    expected_error = re.escape(
        'Unknown dependency step (\'stepA2\',) is in the flow')
    with pytest.raises(Exception, match=expected_error):
        _ = Engine(
            processes={'procB': ProcessB()},
            steps={'stepA1': StepA()},
            topology={
                'stepA1': {'a': ('a',)},
                'procB': {'b': ('b',)},
            },
            flow={'stepA1': [('stepA2',)]},
        )

def test_flow_with_valid_steps() -> None:
    class StepA(Step):

        def ports_schema(self) -> Schema:
            return {
                'a': {'_default': 1}
            }

        def next_update(self, timestep: float, states: State) -> Update:
            return {}

    class ProcessB(Process):

        def ports_schema(self) -> Schema:
            return {
                'b': {'_default': 1}
            }

        def next_update(self, timestep: float, states: State) -> Update:
            return {}

    _ = Engine(
        processes={'procB': ProcessB()},
        steps={
            'stepA1': StepA(),
            'stepA2': StepA(),
        },
        topology={
            'stepA1': {'a': ('a',)},
            'stepA2': {'a': ('a',)},
            'procB': {'b': ('b',)},
        },
        flow={
            'stepA1': [('stepA2',)],
            'stepA2': [],
        },
    )
