"""
============================
Composer and Process Classes
============================
"""

from __future__ import annotations

import abc
import copy
from multiprocessing import Pipe
from multiprocessing import Process as Multiprocess
from multiprocessing.connection import Connection
from typing import (
    Any, Callable, Dict, Optional, Union, List, cast)

from bson.objectid import ObjectId
import numpy as np
from pint.errors import UndefinedUnitError

from vivarium.library.topology import inverse_topology
from vivarium.library.units import Quantity
from vivarium.core.registry import serializer_registry
from vivarium.library.dict_utils import deep_merge, deep_merge_check
from vivarium.core.types import (
    HierarchyPath, Topology, Schema, State, Update, CompositeDict)

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
        processes: Dict[str, Process]) -> None:
    for key, override in overrides.items():
        process = processes[key]
        if isinstance(process, Process):
            process.merge_overrides(override)
        elif isinstance(process, dict):
            _override_schemas(override, process)


def _get_composite_initial_state(
        processes: Dict[str, Process],
        topology: Topology) -> State:
    initial_state = {}
    for path, node in processes.items():
        if isinstance(node, dict):
            for key in node.keys():
                initial_state[key] = _get_composite_initial_state(
                    node, cast(Topology, topology[path]))
        elif isinstance(node, Process):
            process_topology = topology[path]
            process_state = node.initial_state()
            process_path: HierarchyPath = tuple()
            state = inverse_topology(
                process_path, process_state, process_topology)
            initial_state = deep_merge(initial_state, state)

    return initial_state


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
        if config is None:
            config = {}
        if 'name' in config:
            self.name = config['name']
        elif not hasattr(self, 'name'):
            self.name = self.__class__.__name__

        self.config = copy.deepcopy(self.defaults)
        self.config = deep_merge(self.config, config)

        self.merge_processes = self.config.pop('_processes', {})
        self.merge_topology = self.config.pop('_topology', {})
        self.schema_override = self.config.pop('_schema', {})

    def initial_state(self, config: Optional[dict] = None) -> State:
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
        network = self.generate(config)
        processes = cast(Dict[str, Any], network['processes'])
        topology = network['topology']
        initial_state = _get_composite_initial_state(processes, topology)
        return initial_state

    @abc.abstractmethod
    def generate_processes(
            self,
            config: Optional[dict]) -> Dict[str, Any]:
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
        return {}

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
        return {}

    def generate(
            self,
            config: Optional[dict] = None,
            path: HierarchyPath = ()) -> CompositeDict:
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

        processes = self.generate_processes(config)
        topology = self.generate_topology(config)

        # add merged processes
        # TODO - this assumes all merge_processes are initialized.
        # TODO - make option to initialize new processes here
        processes = deep_merge(processes, self.merge_processes)
        topology = deep_merge(topology, self.merge_topology)

        _override_schemas(self.schema_override, processes)

        return {
            'processes': assoc_in({}, path, processes),
            'topology': assoc_in({}, path, topology),
        }

    def get_parameters(self) -> dict:
        """Get the parameters for all :term:`processes`.

        Returns:
            A map from process names to dictionaries of those processes'
            parameters.
        """
        network = self.generate({})
        processes = cast(Dict[str, Process], network['processes'])
        return {
            process_id: process.parameters
            for process_id, process in processes.items()}

    def merge(
            self,
            processes: Optional[Dict[str, Process]] = None,
            topology: Optional[Topology] = None,
            schema_override: Optional[Schema] = None) -> None:
        processes = processes or {}
        topology = topology or {}
        schema_override = schema_override or {}

        for process in processes.values():
            assert isinstance(process, Process)

        self.merge_processes = deep_merge_check(
            self.merge_processes, processes)
        self.merge_topology = deep_merge(self.merge_topology, topology)
        self.schema_override = deep_merge(
            self.schema_override, schema_override)


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
        raise Exception(
            '{} does not include an "initial_state" function'.format(
                self.name))

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
        schema = self.ports_schema()
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
        return {}

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
        return {}

    def or_default(self, parameters: Dict[str, Any], key: str) -> Any:
        """Get parameter from dictionary, falling back to defaults.

        Args:
            parameters: Dictionary to get parameter from.
            key: Parameter key.

        Returns:
            The value of ``key`` from ``parameters``, or the value in
            the class defaults if not in ``parameters``.

        Raises:
            KeyError: If ``key`` is in neither the provided dictionary
                nor the process defaults.
        """
        return parameters.get(key, self.defaults[key])

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
        return {}


class Deriver(Process, metaclass=abc.ABCMeta):
    """Base class for :term:`derivers`."""

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


def test_composite_initial_state() -> None:
    """
    test that initial state in composite merges individual processes'
    initial states
    """
    class AA(Process):
        name = 'AA'

        def initial_state(self, config: Optional[dict] = None) -> State:
            return {'a_port': {'a': 1}}

        def ports_schema(self) -> Schema:
            return {'a_port': {'a': {'_emit': True}}}

        def next_update(
                self,
                timestep: Union[float, int],
                states: State) -> Update:
            return {'a_port': {'a': 1}}

    class BB(Composer):
        name = 'BB'

        def generate_processes(
                self, config: Optional[dict]) -> Dict[str, Any]:
            return {
                'a1': AA({}),
                'a2': AA({}),
                'a3': {
                    'a3_store': AA({})}
            }

        def generate_topology(self, config: Optional[dict]) -> Topology:
            return {
                'a1': {
                    'a_port': ('a1_store',)
                },
                'a2': {
                    'a_port': {
                        'a': ('a1_store', 'b')}
                },
                'a3': {
                    'a3_store': {
                        'a_port': ('a3_1_store',)},
                }
            }

    # run experiment
    bb_composite = BB({})
    initial_state = bb_composite.initial_state()
    expected_initial_state = {
        'a3_store': {
            'a3_1_store': {
                'a': 1}},
        'a1_store': {
            'a': 1,
            'b': 1}}
    assert initial_state == expected_initial_state


class ToyProcess(Process):
    name = 'toy'

    def ports_schema(self) -> Schema:
        return {
            'A': {
                'a': {'_default': 0},
                'b': {'_default': 0}},
            'B': {
                'a': {'_default': 0},
                'b': {'_default': 0}}}

    def next_update(
            self, timestep: Union[float, int], states: State) -> Update:
        return {
            'A': {
                'a': 1,
                'b': states['A']['a']},
            'B': {
                'a': states['A']['b'],
                'b': states['B']['a']}}


class ToyComposite(Composer):
    defaults = {
        'A':  {'name': 'A'},
        'B': {'name': 'B'}}

    def generate_processes(
            self,
            config: Optional[dict]) -> Dict[str, ToyProcess]:
        assert config is not None
        return {
            'A': ToyProcess(config['A']),
            'B': ToyProcess(config['B'])}

    def generate_topology(
            self, config: Optional[dict] = None) -> Topology:
        return {
            'A': {
                'A': ('aaa',),
                'B': ('bbb',)},
            'B': {
                'A': ('bbb',),
                'B': ('ccc',)}}


def test_composite_merge() -> None:
    composer = ToyComposite()
    initial_network = composer.generate()

    expected_initial_topology = {
        'A': {
            'A': ('aaa',),
            'B': ('bbb',),
        },
        'B': {
            'A': ('bbb',),
            'B': ('ccc',),
        },
    }
    assert initial_network['topology'] == expected_initial_topology

    for key in ('A', 'B'):
        assert key in initial_network['processes']
        assert isinstance(initial_network['processes'][key], ToyProcess)

    # merge
    merge_processes: Dict[str, Process] = {
        'C': ToyProcess({'name': 'C'})}
    merge_topology: Topology = {
        'C': {
            'A': ('aaa',),
            'B': ('bbb',)}}
    composer.merge(
        merge_processes,
        merge_topology)

    config = {'A': {'name': 'D'}, 'B': {'name': 'E'}}
    merged_network = composer.generate(config)

    expected_merged_topology = {
        'A': {
            'A': ('aaa',),
            'B': ('bbb',),
        },
        'B': {
            'A': ('bbb',),
            'B': ('ccc',),
        },
        'C': {
            'A': ('aaa',),
            'B': ('bbb',),
        },
    }
    assert merged_network['topology'] == expected_merged_topology

    for key in ('A', 'B', 'C'):
        assert key in merged_network['processes']
        assert isinstance(merged_network['processes'][key], ToyProcess)


def test_get_composite() -> None:
    a = ToyProcess({'name': 'a'})

    a.merge(
        processes={'b': ToyProcess()},
        topology={'b': {
            'A': ('A',),
            'B': ('B',),
        }})

    network = a.generate()

    expected_topology = {
        'a': {
            'A': ('A',),
            'B': ('B',)},
        'b': {
            'A': ('A',),
            'B': ('B',)}}

    assert network['topology'] == expected_topology


if __name__ == '__main__':
    print('Running test_composite_initial_state')
    test_composite_initial_state()
    print('Running test_composite_merge()')
    test_composite_merge()
    print('Running test_get_composite()')
    test_get_composite()
