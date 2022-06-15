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
import os
import pstats
import pickle
from typing import Any, Dict, Optional, Union, List, Tuple
from warnings import warn

import pytest

from vivarium.library.dict_utils import (
    deep_merge, deep_merge_check, deep_copy_internal)
from vivarium.library.topology import assoc_path, get_in
from vivarium.core.types import (
    HierarchyPath, Schema, State, Update,
    Topology, Flow)

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
        processes: Dict[str, 'Process']
) -> None:
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
    """Process parent class.

      All :term:`process` classes must inherit from this class. Each
      class can provide a ``defaults`` class variable to specify the
      process defaults as a dictionary.

      Note that subclasses should call the superclass init function
      first. This allows the superclass to correctly save the initial
      parameters before they are mutated by subclass constructor code.
      We need access to the original parameters for serialization to
      work properly.

      Args:
          parameters: Override the class defaults. This dictionary may
              also contain the following special keys:

              * ``name``: Saved to ``self.name``.
              * ``_original_parameters``: Returned by
                ``__getstate__()`` for serialization.
              * ``_no_original_parameters``: If specified with a value
                of ``True``, original parameters will not be copied
                during initialization, and ``__getstate__()`` will
                instead return ``self.parameters``. This puts the
                responsibility on the user to not mutate process
                parameters.
              * ``_schema``: Overrides the schema.
              * ``_parallel``: Indicates that the process should be
                parallelized. ``self.parallel`` will be set to True.
              * ``_condition``: Path to a variable whose value will be
                returned by
                :py:meth:`vivarium.core.process.Process.update_condition`.
              * ``time_step``: Returned by
                :py:meth:`vivarium.core.process.Process.calculate_timestep`.
      """

    defaults: Dict[str, Any] = {}
    METHOD_COMMANDS = (
        'initial_state', 'generate_processes', 'generate_steps',
        'generate_topology', 'generate_flow', 'merge_overrides',
        'calculate_timestep', 'is_step', 'get_private_state',
        'ports_schema', 'next_update', 'update_condition')
    ATTRIBUTE_READ_COMMANDS = (
        'schema_override', 'parameters', 'condition_path', 'schema')
    ATTRIBUTE_WRITE_COMMANDS = ('set_schema',)

    def __init__(self, parameters: Optional[dict] = None) -> None:
        parameters = parameters or {}
        if '_original_parameters' in parameters:
            original_parameters = parameters.pop('_original_parameters')
        else:
            original_parameters = parameters
        if parameters.get('_no_original_parameters', False):
            self._original_parameters: Optional[dict] = None
        else:
            try:
                self._original_parameters = copy.deepcopy(
                    original_parameters)
            except TypeError:
                # Copying the parameters failed because some parameters do
                # not support being copied.
                self._original_parameters = None

        if 'name' in parameters:
            self.name = parameters['name']
        elif not hasattr(self, 'name'):
            self.name = self.__class__.__name__

        self._parameters = copy.deepcopy(self.defaults)
        self._parameters = deep_merge(self._parameters, parameters)
        self._schema_override: Schema = self._parameters.pop('_schema', {})
        self._parallel = self._parameters.pop('_parallel', False)
        self._condition_path: Optional[HierarchyPath] = None
        self._command_result: Any = None
        self._pending_command: Optional[
            Tuple[str, Optional[tuple], Optional[dict]]] = None

        # set up the conditional state if a condition key is provided
        if '_condition' in self._parameters:
            self._condition_path = self._parameters.pop('_condition')
        if self._condition_path:
            self.merge_overrides(assoc_path({}, self._condition_path, {
                '_default': True,
                '_emit': True,
                '_updater': 'set'}))

        self._set_timestep()

        self._schema: Optional[Schema] = None

    @property
    def parameters(self) -> dict:
        return self._parameters

    @property
    def schema_override(self) -> Schema:
        return self._schema_override

    @property
    def parallel(self) -> bool:
        return self._parallel

    @property
    def condition_path(self) -> Optional[HierarchyPath]:
        return self._condition_path

    @property
    def schema(self) -> Optional[Schema]:
        return self._schema

    @schema.setter
    def schema(self, value: Optional[Schema]) -> None:
        self._schema = value

    def pre_send_command(
            self, command: str, args: Optional[tuple], kwargs:
            Optional[dict]) -> None:
        '''Run pre-checks before starting a command.

        This method should be called at the start of every
        implementation of :py:meth:`send_command`.

        Args:
            command: The name of the command to run.
            args: A tuple of positional arguments for the command.
            kwargs: A dictionary of keyword arguments for the command.

        Raises:
            RuntimeError: Raised when a user tries to send a command
                while a previous command is still pending (i.e. the user
                hasn't called :py:meth:`get_command_result` yet for the
                previous command).
        '''
        if self._pending_command:
            raise RuntimeError(
                f'Trying to send command {(command, args, kwargs)} but '
                f'command {self._pending_command} is still pending.')
        self._pending_command = command, args, kwargs


    def send_command(
            self, command: str, args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            run_pre_check: bool = True) -> None:
        '''Handle :term:`process commands`.

        This method handles the commands listed in
        :py:attr:`METHOD_COMMANDS` by passing ``args``
        and ``kwargs`` to the method of ``self`` with the name
        of the command and saving the return value as the result.

        This method handles the commands listed in
        :py:attr:`ATTRIBUTE_READ_COMMANDS` by returning the attribute of
        ``self`` with the name matching the command, and it handles the
        commands listed in :py:attr:`ATTRIBUTE_WRITE_COMMANDS` by
        setting the attribute in the command to the first argument in
        ``args``. The command must be named ``set_attr`` for attribute
        ``attr``.

        To add support for a custom command, override this function in
        your subclass. Each command is defined by a name (a string)
        and accepts both positional and keyword arguments. Any custom
        commands you add should have associated methods such that:

        * The command name matches the method name.
        * The command and method accept the same positional and keyword
          arguments.
        * The command and method return the same values.

        If all of the above are satisfied, you can use
        :py:meth:`Process.run_command_method` to handle the command.

        Your implementation of this function needs to handle all the
        commands you want to support.  When presented with an unknown
        command, you should call the superclass method, which will
        either handle the command or call its superclass method. At the
        top of this recursive chain, this ``Process.send_command()``
        method handles some built-in commands and will raise an error
        for unknown commands.

        Any overrides of this method must also call
        :py:meth:`pre_send_command` at the start of the method. This
        call will check that no command is currently pending to avoid
        confusing behavior when multiple commands are started without
        intervening retrievals of command results. Since your overriding
        method will have already performed the pre-check, it should pass
        ``run_pre_check=False`` when calling the superclass method.

        Args:
            command: The name of the command to run.
            args: A tuple of positional arguments for the command.
            kwargs: A dictionary of keyword arguments for the command.
            run_pre_check: Whether to run the pre-checks implemented in
                :py:meth:`pre_send_command`. This should be left at its
                default value unless the pre-checks have already been
                performed (e.g. if this method is being called by a
                subclass's overriding method.)

        Returns:
            None. This method just starts the command running.

        Raises:
            ValueError: For unknown commands.
        '''
        if run_pre_check:
            self.pre_send_command(command, args, kwargs)
        args = args or tuple()
        kwargs = kwargs or {}
        if command in self.METHOD_COMMANDS:
            self._command_result = self.run_command_method(
                command, args, kwargs)
        elif command in self.ATTRIBUTE_READ_COMMANDS:
            self._command_result = getattr(self, command)
        elif command in self.ATTRIBUTE_WRITE_COMMANDS:
            assert command.startswith('set_')
            assert args
            setattr(self, command[len('set_'):], args[0])
        else:
            raise ValueError(
                f'Process {self} does not understand the process '
                f'command {command}')

    def run_command_method(
            self, command: str, args: tuple, kwargs: dict) -> Any:
        '''Run a command whose name and interface match a method.

        Args:
            command: The command name, which must equal to a method of
                ``self``.
            args: The positional arguments to pass to the method.
            kwargs: The keywords arguments for the method.

        Returns:
            The result of calling ``self.command(*args, **kwargs)`` is
            returned for command ``command``.
        '''
        return getattr(self, command)(*args, **kwargs)

    def get_command_result(self) -> Any:
        '''Retrieve the result from the last-run command.

        Returns:
            The result of the last command run. Note that this method
            should only be called once immediately after each call to
            :py:meth:`send_command`.

        Raises:
            RuntimeError: When there is no command pending. This can
                happen when this method is called twice without an
                intervening call to :py:meth:`send_command`.
        '''
        if not self._pending_command:
            raise RuntimeError(
                'Trying to retrieve command result, but no command is '
                'pending.')
        self._pending_command = None
        result = self._command_result
        self._command_result = None
        return result

    def run_command(
            self, command: str, args: Optional[tuple] = None,
            kwargs: Optional[dict] = None) -> Any:
        '''Helper function that sends a command and returns result.'''
        self.send_command(command, args, kwargs)
        return self.get_command_result()

    def _set_timestep(self) -> None:
        self._parameters.setdefault('timestep', DEFAULT_TIME_STEP)
        if self._parameters.get('time_step'):
            self._parameters['timestep'] = self._parameters['time_step']

    def __getstate__(self) -> dict:
        """Return parameters

        This is sufficient to reproduce the Process if there are no
        hidden states. Processes with hidden states may need to write
        their own __getstate__.

        The original parameters saved by the constructor are used here,
        so any later changes to the parameters will be lost during
        serialization.
        """
        if self.parameters.get('_no_original_parameters', False):
            return self.parameters
        if self._original_parameters is None:
            raise TypeError(
                'Parameters could not be copied, so serialization is '
                'not supported.')
        return self._original_parameters

    def __setstate__(self, parameters: dict) -> None:
        """Initialize process with parameters"""
        self.__init__(parameters)  # type: ignore

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
        if not self.is_step():
            config = config or {}
            name = config.get('name', self.name)
            return {name: self}
        return {}

    def generate_steps(
            self, config: Optional[dict] = None) -> Dict[str, Any]:
        """Do not override this method."""
        if self.is_step():
            config = config or {}
            name = config.get('name', self.name)
            return {name: self}
        return {}

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

    def generate_flow(self, config: Optional[dict] = None) -> Flow:
        _ = config
        return {}

    def generate(
            self,
            config: Optional[dict] = None,
            path: HierarchyPath = ()) -> Dict:
        if config is None:
            config = self.parameters
        else:
            default = copy.deepcopy(self.parameters)
            config = deep_merge(default, config)
        config = config or {}
        name = config.get('name', self.name)
        processes = self.generate_processes(config)
        steps = self.generate_steps(config)
        topology = self.generate_topology(config)
        flow = self.generate_flow(config)
        processes_and_steps = deep_copy_internal(processes)
        deep_merge_check(processes_and_steps, steps)
        _override_schemas(
            {name: self.schema_override},
            processes_and_steps)

        # TODO -- this should return a Composite instance,
        # but importing Composite introduces circular imports
        # from vivarium.core.composer import Composite
        return {
            'processes': assoc_in({}, path, processes),
            'steps': assoc_in({}, path, steps),
            'topology': assoc_in({}, path, topology),
            'flow': assoc_in({}, path, flow),
            'state': {}
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
        deep_merge(self._schema_override, override)

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

    def calculate_timestep(self, states: Optional[State]) -> Union[float, int]:
        """Return the next process time step

        A process subclass may override this method to implement
        adaptive timesteps. By default it returns self.parameters['timestep'].
        """
        _ = states
        return self.parameters['timestep']

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

        Use the glob '*' schema to declare expected sub-store structure,
        and view all child values of the store:

          .. code-block:: python

            schema = {
                'port1': {
                    '*': {
                        '_default': 1.0
                    }
                }
            }

        Use the glob '**' schema to connect to an entire sub-branch, including
        child nodes, grandchild nodes, etc:

          .. code-block:: python

            schema = {
                'port1': '**'
            }

        Ports flagged as output-only won't be viewed through the next_update's
        states, which can save some overhead time:

          .. code-block:: python

            schema = {
              'port1': {
                 '_output': True,
                 'A': {'_default': 1.0},
              }
            }

        Returns:
            A dictionary that declares which states are expected by the
            processes, and how each state will behave. State keys can be
            assigned properties through schema_keys declared in
            :py:class:`vivarium.core.store.Store`.
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


def _handle_parallel_process(
        connection: Connection, process: Process,
        profile: bool) -> None:
    '''Handle a parallel Vivarium :term:`process`.

    This function is designed to be passed as ``target`` to
    ``Multiprocess()``. In a loop, it receives :term:`process commands`
    from a pipe, passes those commands to the parallel process, and
    passes the result back along the pipe.

    The special command ``end`` is handled directly by this function.
    This command causes the function to exit and therefore shut down the
    OS process created by multiprocessing.

    Args:
        connection: The child end of a multiprocessing pipe. All
            communications received from the pipe should be a 3-tuple of
            the form ``(command, args, kwargs)``, and the tuple contents
            will be passed to :py:meth:`Process.run_command`. The
            result, which may be of any type, will be sent back through
            the pipe.
        process: The process running in parallel.
        profile: Whether to profile the process.
    '''
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
    running = True

    while running:
        command, args, kwargs = connection.recv()

        if command == 'end':
            running = False
        else:
            result = process.run_command(command, args, kwargs)
            connection.send(result)

    if profile:
        profiler.disable()
        stats = pstats.Stats(profiler)
        connection.send(stats.stats)  # type: ignore

    connection.close()


class ParallelProcess(Process):
    def __init__(
            self, process: Process, profile: bool = False,
            stats_objs: Optional[List[pstats.Stats]] = None) -> None:
        """Wraps a :py:class:`Process` for multiprocessing.

        To run a simulation distributed across multiple processors, we
        use Python's multiprocessing tools. This object runs in the main
        process and manages communication between the main (parent)
        process and the child process with the :py:class:`Process` that
        this object manages.

        Most methods pass their name and arguments to
        :py:class:`Process.run_command`.

        Args:
            process: The Process to manage.
            profile: Whether to use cProfile to profile the subprocess.
            stats_objs: List to add cProfile stats objs to when process
                is deleted. Only used if ``profile`` is true.
        """
        super().__init__({
            '_no_original_parameters': True,
            'name': process.name,
            '_parallel': True,
        })
        self.process = process
        self.profile = profile
        self._stats_objs = stats_objs
        assert not self.profile or self._stats_objs is not None
        self.parent, self.child = Pipe()
        self.multiprocess = Multiprocess(
            target=_handle_parallel_process,
            args=(self.child, self.process, self.profile))
        self.multiprocess.start()
        self._ended = False
        self._pending_command: Optional[
            Tuple[str, Optional[tuple], Optional[dict]]] = None

    def send_command(
            self, command: str, args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            run_pre_check: bool = True) -> None:
        '''Send a command to the parallel process.

        See :py:func:``_handle_parallel_process`` for details on how the
        command will be handled.
        '''
        if run_pre_check:
            self.pre_send_command(command, args, kwargs)
        self.parent.send((command, args, kwargs))

    def get_command_result(self) -> Update:
        """Get the result of a command sent to the parallel process.

        Commands and their results work like a queue, so unlike
        :py:class:`Process`, you can technically call this method
        multiple times and get different return values each time.
        This behavior is subject to change, so you should not rely on
        it.

        Returns:
            The command result.
        """
        if not self._pending_command:
            raise RuntimeError(
                'Trying to retrieve command result, but no command is '
                'pending.')
        self._pending_command = None
        return self.parent.recv()

    def initial_state(self, config: Optional[dict] = None) -> State:
        return self.run_command('initial_state', (config,))

    def generate_processes(
            self, config: Optional[dict] = None) -> Dict[str, Any]:
        return self.run_command('generate_processes', (config,))

    def generate_steps(
            self, config: Optional[dict] = None) -> Dict[str, Any]:
        return self.run_command('generate_steps', (config,))

    def generate_topology(
            self, config: Optional[dict] = None) -> Topology:
        return self.run_command('generate_topology', (config,))

    def generate_flow(self, config: Optional[dict] = None) -> Flow:
        return self.run_command('generate_flow', (config,))

    @property
    def schema_override(self) -> Schema:
        return self.run_command('schema_override')

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.run_command('parameters')

    @property
    def condition_path(self) -> Optional[HierarchyPath]:
        return self.run_command('condition_path')

    @property
    def schema(self) -> Schema:
        return self.run_command('schema')

    @schema.setter
    def schema(self, value: Schema) -> None:
        self.run_command('set_schema', (value,))

    def merge_overrides(self, override: Schema) -> None:
        self.run_command('merge_overrides', (override,))

    def calculate_timestep(self, states: Optional[State]) -> float:
        return self.run_command('calculate_timestep', (states,))

    def is_step(self) -> bool:
        return self.run_command('is_step')

    def get_private_state(self) -> State:
        return self.run_command('get_private_state')

    def ports_schema(self) -> Schema:
        return self.run_command('ports_schema')

    def next_update(self, timestep: float, states: State) -> Update:
        return self.run_command('next_update', (timestep, states))

    def update_condition(self, timestep: float, states: State) -> bool:
        return self.run_command('update_condition', (timestep, states))

    def end(self) -> None:
        """End the child process.

        If profiling was enabled, then when the child process ends, it
        will compile its profiling stats and send those to the parent.
        The parent then saves those stats in ``self.stats``.
        """
        # Only end once.
        if self._ended:
            return
        self.send_command('end')
        if self.profile:
            stats = pstats.Stats()
            stats.stats = self.get_command_result()  # type: ignore
            assert self._stats_objs is not None
            self._stats_objs.append(stats)
        self.multiprocess.join()
        self.multiprocess.close()
        self._ended = True

    def __del__(self) -> None:
        self.end()


class ToySerializedProcess(Process):

    defaults: Dict[str, list] = {
        'list': [],
    }

    def __init__(self, parameters: Optional[dict] = None) -> None:
        super().__init__(parameters)
        self.parameters['list'].append(1)

    def ports_schema(self) -> Schema:
        return {}

    def next_update(self, timestep: float, states: State) -> Update:
        return {}


class ToySerializedProcessInheritance(Process):

    defaults = {
        '1': 1,
    }

    def __init__(self, parameters: Optional[dict] = None) -> None:
        parameters = parameters or {}
        super().__init__({
            '2': parameters['1'],
            '_original_parameters': parameters,
        })

    def ports_schema(self) -> Schema:
        return {}

    def next_update(self, timestep: float, states: State) -> Update:
        return {}


class ToyParallelProcess(Process):

    def compare_pid(self, pid: float) -> bool:
        return os.getpid() == pid

    def send_command(
            self, command: str, args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            run_pre_check: bool = True) -> None:
        if run_pre_check:
            self.pre_send_command(command, args, kwargs)
        args = args or tuple()
        kwargs = kwargs or {}
        if command == 'compare_pid':
            self._command_result = self.compare_pid(*args, **kwargs)
        else:
            super().send_command(command, args, kwargs, False)

    def ports_schema(self) -> Schema:
        return {}

    def next_update(self, timestep: float, states: State) -> Update:
        return {}


def test_serialize_process() -> None:
    proc = ToySerializedProcess()
    proc_pickle = pickle.loads(pickle.dumps(proc))

    assert proc.parameters['list'] == [1]
    # If we pickled using `self.parameters` instead of
    # `self._original_parameters`, this list would be [1, 1].
    assert proc_pickle.parameters['list'] == [1]


def test_serialize_process_inheritance() -> None:
    a = ToySerializedProcessInheritance({'1': 0})
    a2 = pickle.loads(pickle.dumps(a))
    assert a2.parameters['2'] == 0


def test_process_commands_pending_safeguard() -> None:
    process = ToySerializedProcess()
    process.send_command('calculate_timestep', (None,))
    with pytest.raises(RuntimeError) as exception:
        process.send_command('next_update', (1, {}))
    msg = "command ('calculate_timestep', (None,), None) is still pending"
    assert msg in str(exception.value)


def test_parallel_process_commands_pending_safeguard() -> None:
    process = ParallelProcess(ToySerializedProcess())
    process.send_command('calculate_timestep', (None,))
    with pytest.raises(RuntimeError) as exception:
        process.send_command('next_update', (1, {}))
    msg = "command ('calculate_timestep', (None,), None) is still pending"
    assert msg in str(exception.value)
    # Reset Process._pending_command so that no warning is thrown when
    # __del__() sends the 'end' command.
    process.get_command_result()


def test_parallel_commands() -> None:
    proc = ToyParallelProcess()
    parallel_proc = ParallelProcess(proc)

    assert proc.compare_pid(os.getpid())
    proc.send_command('compare_pid', (os.getpid(),))
    assert proc.get_command_result()

    parallel_proc.send_command('compare_pid', (os.getpid(),))
    assert not parallel_proc.get_command_result()


def test_invalid_command() -> None:
    proc = ToyParallelProcess()
    with pytest.raises(ValueError) as exception:
        proc.send_command('missing_command')
    msg = 'does not understand the process command missing_command'
    assert msg in str(exception.value)
