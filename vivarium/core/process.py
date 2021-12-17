"""
================
Process Classes
================
"""

import abc
import copy
import cProfile
from multiprocessing import Pipe
from multiprocessing import Process as Multiprocess
from multiprocessing.connection import Connection
import pstats
from typing import (
    Any, Dict, Optional, Union, List)
from warnings import warn

from vivarium.library.dict_utils import deep_merge
from vivarium.library.topology import assoc_path, get_in
from vivarium.core.types import (
    HierarchyPath, Schema, State, Update,
    Topology)

DEFAULT_TIME_STEP = 1.0


def assoc_in(d: dict, path: HierarchyPath, value: Any) -> dict:
    """Insert a value into a dictionary at an arbitrary depth.

    Empty dictionaries will be created as needed to insert the value at
    the specified depth.

    >>> d = {'a': {'b': 1}}
    >>> assoc_in(d, ('a', 'c', 'd'), 2)
    {'a': {'b': 1, 'c': {'d': 2}}}

    Args:
        d: Dictionary to insert into.
        path: Path in the dictionary where the value will be inserted.
            Each element of the path is dictionary key, which will be
            added if not already present. Any given element (except the
            first) refers to a key in the dictionary that is the value
            associated with the immediately preceding path element.
        value: The value to insert.

    Returns:
        Dictionary with the value inserted.
    """
    if path:
        return dict(
            d,
            **{
                path[0]: assoc_in(d.get(path[0], {}), path[1:], value)
            }
        )
    return value


def _override_schemas(
        overrides: Dict[str, Schema],
        processes: Dict[str, 'Process']) -> None:
    for key, override in overrides.items():
        process = processes[key]
        if isinstance(process, Process):
            process.merge_overrides(override)
        elif isinstance(process, dict):
            _override_schemas(override, process)


def _get_parameters(
        processes: Optional[Dict[str, 'Process']] = None
) -> Dict:
    processes = processes or {}
    parameters: Dict = {}
    for key, value in processes.items():
        if isinstance(value, Process):
            parameters[key] = value.parameters
        elif isinstance(value, dict):
            parameters[key] = _get_parameters(value)
    return parameters


class Process(metaclass=abc.ABCMeta):
    defaults: Dict[str, Any] = {}

    def __init__(self, parameters: Optional[dict] = None) -> None:
        """Process parent class.

        All :term:`process` classes must inherit from this class. Each
        class can provide a ``defaults`` class variable to specify the
        process defaults as a dictionary.

        Args:
            parameters: Override the class defaults. If this contains a
                ``_register`` key, the process will register itself in
                the process registry.
        """
        parameters = parameters or {}
        if 'name' in parameters:
            self.name = parameters['name']
        elif not hasattr(self, 'name'):
            self.name = self.__class__.__name__

        self.parameters = copy.deepcopy(self.defaults)
        self.parameters = deep_merge(self.parameters, parameters)
        self.schema_override = self.parameters.pop('_schema', {})
        self.parallel = self.parameters.pop('_parallel', False)
        self.condition_path = None

        # set up the conditional state if a condition key is provided
        if '_condition' in self.parameters:
            self.condition_path = self.parameters.pop('_condition')
        if self.condition_path:
            self.merge_overrides(assoc_path({}, self.condition_path, {
                '_default': True,
                '_emit': True,
                '_updater': 'set'}))

        self.parameters.setdefault('time_step', DEFAULT_TIME_STEP)
        self.schema: Optional[dict] = None

    def __getstate__(self) -> dict:
        """Return parameters

        This is sufficient to reproduce the Process if there are no
        hidden states. Processes with hidden states may need to write
        their own __getstate__.
        """
        return self.parameters

    def __setstate__(self, state: dict) -> None:
        """Initialize process with parameters"""
        self.__init__(parameters=state)  # type: ignore

    def initial_state(self, config: Optional[dict] = None) -> State:
        """Get initial state in embedded path dictionary.

        Every subclass may override this method.

        Args:
            config (dict): A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            dict: Subclass implementations must return a dictionary
            mapping state paths to initial values.
        """
        _ = config
        return {}

    def generate_processes(
            self, config: Optional[dict] = None) -> Dict[str, Any]:
        """Do not override this method."""
        config = config or {}
        name = config.get('name', self.name)
        return {name: self}

    def generate_topology(self, config: Optional[dict] = None) -> Topology:
        """Do not override this method."""
        config = config or {}
        name = config.get('name', self.name)
        override_topology = config.get('topology', {})
        ports = self.ports()
        return {
            name: {
                port: override_topology.get(port, (port,))
                for port in ports.keys()}}

    def generate(
            self,
            config: Optional[dict] = None,
            path: HierarchyPath = ()) -> Dict:
        if config is None:
            config = self.parameters
        else:
            default = copy.deepcopy(self.parameters)
            config = deep_merge(default, config)

        processes = self.generate_processes(config)
        topology = self.generate_topology(config)
        _override_schemas(self.schema_override, processes)

        return {
            'processes': assoc_in({}, path, processes),
            'topology': assoc_in({}, path, topology),
        }

    def get_schema(self, override: Optional[Schema] = None) -> dict:
        """Get the process's schema, optionally with a schema override.

        Args:
            override: Override schema

        Returns:
            The combined schema.
        """
        ports = copy.deepcopy(self.ports_schema())
        deep_merge(ports, self.schema_override)
        deep_merge(ports, override)
        return ports

    def merge_overrides(self, override: Schema) -> None:
        """Add a schema override to the process's schema overrides.

        Args:
            override: The schema override to add.
        """
        deep_merge(self.schema_override, override)

    def ports(self) -> Dict[str, List[str]]:
        """Get ports and each port's variables.

        Returns:
            A map from port names to lists of the variables that go into
            that port.
        """
        ports_schema = self.ports_schema()
        return {
            port: list(states.keys())
            for port, states in ports_schema.items()}

    def local_timestep(self) -> Union[float, int]:
        """Get a process's favored timestep.

        The timestep may change over the course of the simulation.
        Processes must not assume that their favored timestep will
        actually be used. To customize their timestep, processes can
        override this method.

        Returns:
            Favored timestep.
        """
        return self.parameters['time_step']

    def calculate_timestep(self, states: Optional[State]) -> Union[float, int]:
        """Return the next process time step

        A process subclass may override this method to implement
        adaptive timesteps. By default it returns self.parameters['time_step'].
        """
        _ = states
        return self.parameters['time_step']

    def default_state(self) -> State:
        """Get the default values of the variables in each port.

        The default values are computed based on the schema.

        Returns:
            A state dictionary that assigns each variable's default
            value to that variable.
        """
        schema = self.get_schema()
        state: State = {}
        for port, states in schema.items():
            for key, value in states.items():
                if '_default' in value:
                    if port not in state:
                        state[port] = {}
                    state[port][key] = value['_default']
        return state

    def is_deriver(self) -> bool:
        """Check whether this process is a :term:`deriver`.

        .. deprecated:: 0.3.14
           Derivers have been deprecated in favor of :term:`steps`, so
           please override ``is_step`` instead of ``is_deriver``.
           Support for Derivers may be removed in a future release.

        Returns:
            Whether this process is a deriver. This class always returns
            ``False``, but subclasses may change this.
        """
        return False

    def is_step(self) -> bool:
        """Check whether this process is a :term:`step`.

        Returns:
            Whether this process is a step. This class always returns
            ``False``, but subclasses may change this behavior.
        """
        method = getattr(self.is_deriver, '__func__', None)
        if method and method is not Process.is_deriver:
            # `self` is an instance of a subclass that has overridden
            # `is_deriver`.
            warn(
                'is_deriver() is deprecated. Use is_step() instead.',
                category=FutureWarning)
            return self.is_deriver()
        return False

    def get_private_state(self) -> State:
        """Get the process's private state.

        Processes can store state in instance variables instead of in
        the :term:`stores` that hold the simulation-wide state. These
        instance variables hold a private state that is not shared with
        other processes. You can override this function to let
        :term:`experiments` emit the process's private state.

        Returns:
            An empty dictionary. You may override this behavior to
            return your process's private state.
        """
        return {}  # pragma: no cover

    @abc.abstractmethod
    def ports_schema(self) -> Schema:
        """Get the schemas for each port.

        This must be overridden by any subclasses.

        Returns:
            A dictionary that declares which states are expected by the
            processes, and how each state will behave. State keys can be
            assigned properties through schema_keys declared in
            :py:class:`vivarium.core.store.Store`. Ports flagged with
            {'_output': True} make it an output-only port, that won't
            be viewed through the next_update's states.
        """
        return {}  # pragma: no cover

    @abc.abstractmethod
    def next_update(
            self, timestep: Union[float, int], states: State) -> Update:
        """Compute the next update to the simulation state.

        Args:
            timestep: The duration for which the update should be
                computed.
            states: The pre-update simulation state. This will take the
                same form as the process's schema, except with a value
                for each variable.

        Returns:
            An empty dictionary for now. This should be overridden by
            each subclass to return an update.
        """
        return {}  # pragma: no cover

    def update_condition(
            self, timestep: Union[float, int], states: State) -> Any:
        """Determine whether this process runs.

        Args:
            timestep: The duration for which an update.
            states: The pre-update simulation state.

        Returns:
            Boolean for whether this process runs. True by default.
        """

        _ = timestep
        _ = states

        # use the given condition key if it was provided
        if self.condition_path:
            return get_in(states, self.condition_path)

        return True


class Step(Process, metaclass=abc.ABCMeta):
    """Base class for steps."""

    def __init__(self, parameters: Optional[dict] = None) -> None:
        parameters = parameters or {}
        super().__init__(parameters)

    def is_step(self) -> bool:
        """Returns ``True`` to signal that this process is a step."""
        return True


#: Deriver is just an alias for :py:class:`Step` now that Derivers have
#: been deprecated.
Deriver = Step


def _run_update(
        connection: Connection, process: Process,
        profile: bool) -> None:
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
    running = True

    while running:
        interval, states = connection.recv()

        # stop process by sending -1 as the interval
        if interval == -1:
            running = False

        else:
            update = process.next_update(interval, states)
            connection.send(update)

    if profile:
        profiler.disable()
        stats = pstats.Stats(profiler)
        connection.send(stats.stats)  # type: ignore

    connection.close()


class ParallelProcess:
    def __init__(self, process: Process, profile: bool = False) -> None:
        """Wraps a :py:class:`Process` for multiprocessing.

        To run a simulation distributed across multiple processors, we
        use Python's multiprocessing tools. This object runs in the main
        process and manages communication between the main (parent)
        process and the child process with the :py:class:`Process` that
        this object manages.

        Args:
            process: The Process to manage.
            profile: Whether to use cProfile to profile the subprocess.
        """
        self.process = process
        self.profile = profile
        self.stats: Optional[pstats.Stats] = None
        self.parent, self.child = Pipe()
        self.multiprocess = Multiprocess(
            target=_run_update,
            args=(self.child, self.process, self.profile))
        self.multiprocess.start()

    def update(
            self, interval: Union[float, int], states: State) -> None:
        """Request an update from the process.

        Args:
            interval: The length of the timestep for which the update
                should be computed.
            states: The pre-update state of the simulation.
        """
        self.parent.send((interval, states))

    def get(self) -> Update:
        """Get an update from the process.

        Returns:
            The update from the process.
        """
        return self.parent.recv()

    def end(self) -> None:
        """End the child process.

        If profiling was enabled, then when the child process ends, it
        will compile its profiling stats and send those to the parent.
        The parent then saves those stats in ``self.stats``.
        """
        self.parent.send((-1, None))
        if self.profile:
            self.stats = pstats.Stats()
            self.stats.stats = self.parent.recv()  # type: ignore
        self.multiprocess.join()
