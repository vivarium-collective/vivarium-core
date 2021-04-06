"""
============================
Composer and Process Classes
============================
"""

import abc
import copy
from multiprocessing import Pipe
from multiprocessing import Process as Multiprocess
from multiprocessing.connection import Connection
from typing import (
    Any, Callable, Dict, Optional, Union, List, Iterable, cast)

from bson.objectid import ObjectId
import numpy as np
from pint.errors import UndefinedUnitError

from vivarium.library.datum import Datum
from vivarium.library.topology import (
    inverse_topology, convert_path_to_tuple, convert_topology_path)
from vivarium.library.units import Quantity, Unit
from vivarium.core.registry import serializer_registry
from vivarium.library.dict_utils import deep_merge
from vivarium.core.types import (
    HierarchyPath, TuplePath, Topology,
    Schema, State, Update, Processes)

DEFAULT_TIME_STEP = 1.0


def serialize_value(value: Any) -> Any:
    """Attempt to serialize a value.

    For this function, consider "serializable" to mean serializiable by
    this function.  This function can serialize the following kinds of
    values:

    * :py:class:`dict` whose keys implement ``__str__`` and whose values
      are serializable. Keys are serialized by calling ``__str__`` and
      values are serialized by this function.
    * :py:class:`list` and :py:class:`tuple`, whose values are
      serializable. The value is serialized as a list of the serialized
      values.
    * Numpy ndarray objects are handled by
      :py:class:`vivarium.core.registry.NumpySerializer`.
    * :py:class:`pint.Quantity` objects are handled by
      :py:class:`vivarium.core.registry.UnitsSerializer`.
    * Functions are handled by
      :py:class:`vivarium.core.registry.FunctionSerializer`.
    * :py:class:`vivarium.core.process.Process` objects are handled by
      :py:class:`vivarium.core.registry.ProcessSerializer`.
    * :py:class:`vivarium.core.process.Composer` objects are handled by
      :py:class:`vivarium.core.registry.ComposerSerializer`.
    * Numpy scalars are handled by
      :py:class:`vivarium.core.registry.NumpyScalarSerializer`.
    * ``ObjectId`` objects are serialized by calling its ``__str__``
      function.

    When provided with a serializable value, the returned serialized
    value is suitable for inclusion in a JSON object.

    Args:
        value: The value to serialize.

    Returns:
        The serialized value if ``value`` is serializable. Otherwise,
        ``value`` is returned unaltered.
    """
    if isinstance(value, dict):
        value = cast(dict, value)
        return _serialize_dictionary(value)
    if isinstance(value, list):
        value = cast(list, value)
        return _serialize_list(value)
    if isinstance(value, tuple):
        value = cast(tuple, value)
        return _serialize_list(list(value))
    if isinstance(value, np.ndarray):
        value = cast(np.ndarray, value)
        return serializer_registry.access('numpy').serialize(value)
    if isinstance(value, Quantity):
        value = cast(Quantity, value)
        return serializer_registry.access('units').serialize(value)
    if isinstance(value, Unit):
        value = cast(Unit, value)
        return serializer_registry.access('units').serialize(value)
    if callable(value):
        value = cast(Callable, value)
        return serializer_registry.access('function').serialize(value)
    if isinstance(value, Process):
        value = cast(Process, value)
        return _serialize_dictionary(
            serializer_registry.access('process').serialize(value))
    if isinstance(value, Composer):
        value = cast(Composer, value)
        return _serialize_dictionary(
            serializer_registry.access('composer').serialize(value))
    if isinstance(value, (np.integer, np.floating)):
        return serializer_registry.access(
            'numpy_scalar').serialize(value)
    if isinstance(value, ObjectId):
        value = cast(ObjectId, value)
        return str(value)
    return value


def deserialize_value(value: Any) -> Any:
    """Attempt to deserialize a value.

    Supports deserializing the following kinds ov values:

    * :py:class:`dict` with serialized values and keys that need not be
      deserialized. The values will be deserialized with this function.
    * :py:class:`list` of serialized values. The values will be
      deserialized with this function.
    * :py:class:`str` which are serialized :py:class:`pint.Quantity`
      values.  These will be deserialized with
      :py:class:`vivarium.core.registry.UnitsSerializer`.

    Args:
        value: The value to deserialize.

    Returns:
        The deserialized value if ``value`` is of a supported type.
        Otherwise, returns ``value`` unmodified.
    """
    if isinstance(value, dict):
        value = cast(dict, value)
        return _deserialize_dictionary(value)
    if isinstance(value, list):
        value = cast(list, value)
        return _deserialize_list(value)
    if isinstance(value, str):
        value = cast(str, value)
        try:
            return serializer_registry.access(
                'units').deserialize(value)
        except UndefinedUnitError:
            return value
    return value


def _serialize_list(lst: list) -> list:
    serialized = []
    for value in lst:
        serialized.append(serialize_value(value))
    return serialized


def _serialize_dictionary(d: dict) -> Dict[str, Any]:
    serialized = {}
    for key, value in d.items():
        if not isinstance(key, str):
            key = str(key)
        serialized[key] = serialize_value(value)
    return serialized


def _deserialize_list(lst: list) -> list:
    deserialized = []
    for value in lst:
        deserialized.append(deserialize_value(value))
    return deserialized


def _deserialize_dictionary(d: dict) -> dict:
    deserialized = {}
    for key, value in d.items():
        deserialized[key] = deserialize_value(value)
    return deserialized


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


def _get_composite_state(
        processes: Dict[str, 'Process'],
        topology: Any,
        state_type: Optional[str] = 'initial',
        path: Optional[TuplePath] = None,
        initial_state: Optional[State] = None,
        config: Optional[dict] = None,
) -> Optional[State]:

    path = path or tuple()
    initial_state = initial_state or {}
    config = config or {}

    for key, node in processes.items():
        subpath = path + (key,)
        subtopology = topology[key]

        if isinstance(node, dict):
            state = _get_composite_state(
                processes=node,
                topology=subtopology,
                state_type=state_type,
                path=subpath,
                initial_state=initial_state,
                config=config.get(key),
            )
        elif isinstance(node, Process):
            if state_type == 'initial':
                # get the initial state
                process_state = node.initial_state(config.get(node.name))
            elif state_type == 'default':
                # get the default state
                process_state = node.default_state()
            state = inverse_topology(path, process_state, subtopology)

        initial_state = deep_merge(initial_state, state)

    return initial_state


class Composite(Datum):
    """Composite parent class.

    Contains keys for processes and topology
    """
    processes: Dict[str, Any] = {}
    topology: Dict[str, Any] = {}
    _schema: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {
        'processes': processes,
        'topology': topology,
        '_schema': _schema}

    def __init__(
            self,
            config: Dict[str, Any]
    ) -> None:
        super().__init__(config)
        _override_schemas(self._schema, self.processes)

    def initial_state(self, config: Optional[dict] = None) -> Optional[State]:
        """ Merge all processes' initial states
        Arguments:
            config (dict): A dictionary of configuration options. All
            subclass implementation must accept this parameter, but
            some may ignore it.
        Returns:
            (dict): Subclass implementations must return a dictionary
            mapping state paths to initial values.
        """
        return _get_composite_state(
            processes=self.processes,
            topology=self.topology,
            state_type='initial',
            config=config)

    def default_state(self, config: Optional[dict] = None) -> Optional[State]:
        """ Merge all processes' default states
        Arguments:
            config (dict): A dictionary of configuration options. All
            subclass implementation must accept this parameter, but
            some may ignore it.
        Returns:
            (dict): Subclass implementations must return a dictionary
            mapping state paths to default values.
        """
        return _get_composite_state(
            processes=self.processes,
            topology=self.topology,
            state_type='default',
            config=config)

    def merge(
            self,
            composite: Optional['Composite'] = None,
            processes: Optional[Dict[str, 'Process']] = None,
            topology: Optional[Topology] = None,
            path: Optional[HierarchyPath] = None,
            schema_override: Optional[Schema] = None,
    ) -> None:
        composite = composite or Composite({})
        processes = processes or {}
        topology = topology or {}
        path = path or tuple()
        path = convert_path_to_tuple(path)
        schema_override = schema_override or {}

        # get the processes and topology to merge
        merge_processes = {}
        merge_topology = {}
        if composite:
            merge_processes.update(composite['processes'])
            merge_topology.update(composite['topology'])
        deep_merge(merge_processes, processes)
        deep_merge(merge_topology, topology)
        merge_processes = assoc_in({}, path, merge_processes)
        merge_topology = assoc_in({}, path, merge_topology)

        # merge with instance processes and topology
        deep_merge(self.processes, merge_processes)
        deep_merge(self.topology, merge_topology)
        self._schema.update(schema_override)
        _override_schemas(self._schema, self.processes)

    def get_parameters(self) -> Dict:
        """Get the parameters for all :term:`processes`.
        Returns:
            A map from process names to parameters.
        """
        return _get_parameters(self.processes)


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


class Composer(metaclass=abc.ABCMeta):
    defaults: Dict[str, Any] = {}

    def __init__(self, config: Optional[dict] = None) -> None:
        """Base class for :term:`composer` classes.

        Composers generate :term:`composites`.

        All :term:`composer` classes must inherit from this class.

        Args:
            config: Dictionary of configuration options that can
                override the class defaults.
        """
        config = config or {}
        if 'name' in config:
            self.name = config['name']
        elif not hasattr(self, 'name'):
            self.name = self.__class__.__name__

        self.config = copy.deepcopy(self.defaults)
        self.config = deep_merge(self.config, config)
        self.schema_override = self.config.pop('_schema', {})

    @abc.abstractmethod
    def generate_processes(
            self,
            config: Optional[dict]) -> Processes:
        """Generate processes dictionary.

        Every subclass must override this method.

        Args:
            config: A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            Subclass implementations must return a dictionary
            mapping process names to instantiated and configured process
            objects.
        """
        return {}  # pragma: no cover

    @abc.abstractmethod
    def generate_topology(self, config: Optional[dict]) -> Topology:
        """Generate topology dictionary.

        Every subclass must override this method.

        Args:
            config: A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            Subclass implementations must return a :term:`topology`
            dictionary.
        """
        return {}  # pragma: no cover

    def generate(
            self,
            config: Optional[dict] = None,
            path: HierarchyPath = '') -> Composite:
        """Generate processes and topology dictionaries.

        Args:
            config: Updates values in the configuration declared
                in the constructor.
            path: Tuple with ('path', 'to', 'level') associates
                the processes and topology at this level.

        Returns:
            Dictionary with keys ``processes``, which has a value of a
            processes dictionary, and ``topology``, which has a value of
            a topology dictionary. Both are suitable to be passed to the
            constructor for
            :py:class:`vivarium.core.experiment.Experiment`.
        """
        if config is None:
            config = self.config
        else:
            default = copy.deepcopy(self.config)
            config = deep_merge(default, config)
        path = convert_path_to_tuple(path)

        processes = self.generate_processes(config)
        topology = self.generate_topology(config)
        topology = convert_topology_path(topology)
        _override_schemas(self.schema_override, processes)

        return Composite({
            'processes': assoc_in({}, path, processes),
            'topology': assoc_in({}, path, topology),
        })

    def initial_state(self, config: Optional[dict] = None) -> Optional[State]:
        """ Merge all processes' initial states

        Every subclass may override this method.

        Arguments:
            config (dict): A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            dict: Subclass implementations must return a dictionary
            mapping state paths to initial values.
        """
        composite = self.generate(config)
        return composite.initial_state(config)

    def get_parameters(self) -> dict:
        """Get the parameters for all :term:`processes`.

        Returns:
            A map from process names to dictionaries of those processes'
            parameters.
        """
        composite = self.generate()
        return composite.get_parameters()


class MetaComposer(Composer):

    def __init__(
            self,
            composers: Iterable[Any] = tuple(),
            config: Optional[dict] = None,
    ) -> None:
        super().__init__(config)
        self.composers: List = list(composers)

    def generate_processes(
            self,
            config: Optional[dict] = None
    ) -> Dict[str, Any]:
        # TODO(Eran)-- override composite.config with config
        processes: Dict = {}
        for composer in self.composers:
            new_processes = composer.generate_processes(composer.config)
            if set(processes.keys()) & set(new_processes.keys()):
                raise ValueError(
                    f"{set(processes.keys())} and {set(new_processes.keys())} "
                    f"in contain overlapping keys")
            processes.update(new_processes)
        return processes

    def generate_topology(
            self,
            config: Optional[dict] = None
    ) -> Topology:
        topology: Topology = {}
        for composer in self.composers:
            new_topology = composer.generate_topology(composer.config)
            if set(topology.keys()) & set(new_topology.keys()):
                raise ValueError(
                    f"{set(topology.keys())} and {set(new_topology.keys())} "
                    f"contain overlapping keys")
            topology.update(new_topology)
        return topology

    def add_composer(
            self,
            composer: Composer,
            config: Optional[Dict] = None,
    ) -> None:
        if config:
            self.config.update(config)
        self.composers.append(composer)

    def add_composers(
            self,
            composers: List,
            config: Optional[Dict] = None,
    ) -> None:
        if config:
            self.config.update(config)
        self.composers.extend(composers)


class Process(Composer, metaclass=abc.ABCMeta):
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
        super().__init__(parameters)

        self.parameters = self.config
        self.parallel = self.config.pop('_parallel', False)
        self.parameters.setdefault('time_step', DEFAULT_TIME_STEP)
        self.schema = None

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
        return {}

    def generate_processes(
            self, config: Optional[dict] = None) -> Dict[str, Any]:
        config = config or {}
        name = config.get('name', self.name)
        return {name: self}

    def generate_topology(self, config: Optional[dict] = None) -> Topology:
        config = config or {}
        name = config.get('name', self.name)
        override_topology = config.get('topology', {})
        ports = self.ports()
        return {
            name: {
                port: override_topology.get(port, (port,))
                for port in ports.keys()}}

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
        """Check whether this process is a deriver.

        Returns:
            Whether this process is a deriver. This class always returns
            ``False``, but subclasses may change this.
        """
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


class Deriver(Process, metaclass=abc.ABCMeta):
    """Base class for :term:`derivers`."""
    def initial_state(self, config: Optional[dict] = None) -> State:
        return {}

    def is_deriver(self) -> bool:
        """Returns ``True`` to signal this process is a deriver."""
        return True


def _run_update(connection: Connection, process: Process) -> None:
    running = True

    while running:
        interval, states = connection.recv()

        # stop process by sending -1 as the interval
        if interval == -1:
            running = False

        else:
            update = process.next_update(interval, states)
            connection.send(update)

    connection.close()


class ParallelProcess:
    def __init__(self, process: Process) -> None:
        """Wraps a :py:class:`Process` for multiprocessing.

        To run a simulation distributed across multiple processors, we
        use Python's multiprocessing tools. This object runs in the main
        process and manages communication between the main (parent)
        process and the child process with the :py:class:`Process` that
        this object manages.

        Args:
            process: The Process to manage.
        """
        self.process = process
        self.parent, self.child = Pipe()
        self.multiprocess = Multiprocess(
            target=_run_update,
            args=(self.child, self.process))
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
        """End the child process."""
        self.parent.send((-1, None))
        self.multiprocess.join()
