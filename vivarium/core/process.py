"""
==========================================
Factory, Process, and Composite Classes
==========================================
"""

import abc
import copy
from multiprocessing import Pipe
from multiprocessing import Process as Multiprocess
from multiprocessing.connection import Connection
from typing import (
    Any, Callable, Dict, Optional, Tuple, Union, List, cast)

from bson.objectid import ObjectId
import numpy as np
from pint.errors import UndefinedUnitError

from vivarium.library.topology import inverse_topology
from vivarium.library.units import Quantity
from vivarium.core.registry import process_registry, serializer_registry
from vivarium.library.dict_utils import deep_merge, deep_merge_check
from vivarium.core.types import (
    Path, Topology, Schema, State, Update, CompositeDict)

DEFAULT_TIME_STEP = 1.0


def serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        value = cast(dict, value)
        return serialize_dictionary(value)
    if isinstance(value, list):
        value = cast(list, value)
        return serialize_list(value)
    if isinstance(value, tuple):
        value = cast(tuple, value)
        return serialize_list(list(value))
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
        return serialize_dictionary(
            serializer_registry.access('process').serialize(value))
    if isinstance(value, Factory):
        value = cast(Factory, value)
        return serialize_dictionary(
            serializer_registry.access('factory').serialize(value))
    if isinstance(value, (np.integer, np.floating)):
        value = cast(Union[np.integer, np.floating], value)
        return serializer_registry.access(
            'numpy_scalar').serialize(value)
    if isinstance(value, ObjectId):
        value = cast(ObjectId, value)
        return str(value)
    return value


def deserialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        value = cast(dict, value)
        return deserialize_dictionary(value)
    if isinstance(value, list):
        value = cast(list, value)
        return deserialize_list(value)
    if isinstance(value, str):
        value = cast(str, value)
        try:
            return serializer_registry.access(
                'units').deserialize(value)
        except UndefinedUnitError:
            return value
    return value


def serialize_list(lst: list) -> list:
    serialized = []
    for value in lst:
        serialized.append(serialize_value(value))
    return serialized


def serialize_dictionary(d: dict) -> Dict[str, Any]:
    serialized = {}
    for key, value in d.items():
        if not isinstance(key, str):
            key = str(key)
        serialized[key] = serialize_value(value)
    return serialized


def deserialize_list(lst: list) -> list:
    deserialized = []
    for value in lst:
        deserialized.append(deserialize_value(value))
    return deserialized


def deserialize_dictionary(d: dict) -> dict:
    deserialized = {}
    for key, value in d.items():
        deserialized[key] = deserialize_value(value)
    return deserialized


def assoc_in(d: dict, path: Path, value: Any) -> dict:
    if path:
        return dict(
            d,
            **{
                path[0]: assoc_in(d.get(path[0], {}), path[1:], value)
            }
        )
    return value


def override_schemas(
        overrides: Dict[str, Schema],
        processes: Dict[str, 'Process']) -> None:
    for key, override in overrides.items():
        process = processes[key]
        if isinstance(process, Process):
            process.merge_overrides(override)
        else:
            override_schemas(override, process)


def generate_derivers(
        processes: Dict[str, 'Process'],
        topology: Topology) -> CompositeDict:
    deriver_processes = {}
    deriver_topology = Topology({})
    for process_key, node in processes.items():
        subtopology = topology[process_key]
        if isinstance(node, Process):
            for deriver_key, config in node.derivers().items():
                if deriver_key not in deriver_processes:
                    # generate deriver process
                    deriver_config = config.get('config', {})
                    generate = config['deriver']
                    if isinstance(generate, str):
                        generate = process_registry.access(generate)

                    deriver = generate(deriver_config)
                    deriver_processes[deriver_key] = deriver

                    # generate deriver topology
                    deriver_ports = deriver.ports()
                    deriver_topology[deriver_key] = {
                        port: (port,) for port in deriver_ports.keys()}
                    for target, source in config.get(
                            'port_mapping', {}).items():
                        path = subtopology[source]
                        deriver_topology[deriver_key][target] = path
        else:
            subderivers = generate_derivers(node, subtopology)
            deriver_processes[process_key] = subderivers['processes']
            deriver_topology[process_key] = subderivers['topology']
    return CompositeDict({
        'processes': deriver_processes,
        'topology': deriver_topology})


def get_composite_initial_state(
        processes: Dict[str, 'Process'],
        topology: Topology) -> State:
    initial_state = {}
    for path, node in processes.items():
        if isinstance(node, dict):
            for key in node.keys():
                initial_state[key] = get_composite_initial_state(
                    node, topology[path])
        elif isinstance(node, Process):
            process_topology = topology[path]
            process_state = node.initial_state()
            process_path: Path = tuple()
            state = inverse_topology(
                process_path, process_state, process_topology)
            initial_state = deep_merge(initial_state, state)

    return State(initial_state)


class Factory(metaclass=abc.ABCMeta):
    """Factory parent class

    All :term:`factory` classes must inherit from this class.
    """
    defaults: Dict[str, Any] = {}

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        if 'name' in config:
            self.name = config['name']
        elif not hasattr(self, 'name'):
            self.name = self.__class__.__name__

        self.config = copy.deepcopy(self.defaults)
        self.config = deep_merge(self.config, config)

    @abc.abstractmethod
    def generate_processes(
            self,
            config: Optional[dict]) -> Dict[str, Any]:
        """Generate processes dictionary

        Every subclass must override this method.

        Arguments:
            config (dict): A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            dict: Subclass implementations must return a dictionary
            mapping process names to instantiated and configured process
            objects.
        """
        return {}

    @abc.abstractmethod
    def generate_topology(self, config: Optional[dict]) -> Topology:
        """Generate topology dictionary

        Every subclass must override this method.

        Arguments:
            config (dict): A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            dict: Subclass implementations must return a
            :term:`topology` dictionary.
        """
        return Topology({})

    def generate(
            self,
            config: Optional[dict] = None,
            path: Path = tuple()) -> CompositeDict:
        '''Generate processes and topology dictionaries

        Arguments:
            config (dict): Updates values in the configuration declared
                in the constructor
            path (tuple): Tuple with ('path', 'to', 'level') associates
                the processes and topology at this level

        Returns:
            dict: Dictionary with two keys: ``processes``, which has a
            value of a processes dictionary, and ``topology``, which has
            a value of a topology dictionary. Both are suitable to be
            passed to the constructor for
            :py:class:`vivarium.core.experiment.Experiment`.
        '''

        if config is None:
            config = self.config
        else:
            default = copy.deepcopy(self.config)
            config = deep_merge(default, config)

        processes = self.generate_processes(config)
        topology = self.generate_topology(config)

        # add derivers
        derivers = generate_derivers(processes, topology)
        processes = deep_merge(derivers['processes'], processes)
        topology = deep_merge(derivers['topology'], topology)

        return CompositeDict({
            'processes': assoc_in({}, path, processes),
            'topology': Topology(assoc_in({}, path, topology)),
        })


class Composite(Factory):
    """Composite parent class

    All :term:`composite` classes must inherit from this class.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

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
        processes = network['processes']
        topology = cast(Topology, network['topology'])
        initial_state = get_composite_initial_state(processes, topology)
        return initial_state

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return Topology({})

    def generate(
            self,
            config: Optional[dict] = None,
            path: Path = tuple()) -> CompositeDict:
        network = super().generate(config=config)
        processes = network['processes']
        topology = network['topology']

        # add merged processes
        # TODO - this assumes all merge_processes are initialized.
        # TODO - make option to initialize new processes here
        processes = deep_merge(processes, self.merge_processes)
        topology = deep_merge(topology, self.merge_topology)

        override_schemas(self.schema_override, processes)

        return CompositeDict({
            'processes': assoc_in({}, path, processes),
            'topology': Topology(assoc_in({}, path, topology)),
        })

    def get_parameters(self) -> dict:
        network = self.generate({})
        processes = network['processes']
        return {
            process_id: process.parameters
            for process_id, process in processes.items()}

    def merge(
            self,
            processes: Optional[Dict[str, 'Process']] = None,
            topology: Optional[Topology] = None,
            schema_override: Optional[Schema] = None) -> None:
        processes = processes or {}
        topology = topology or Topology({})
        schema_override = schema_override or Schema({})

        for process in processes.values():
            assert isinstance(process, Process)

        self.merge_processes = deep_merge_check(
            self.merge_processes, processes)
        self.merge_topology = deep_merge(self.merge_topology, topology)
        self.schema_override = deep_merge(
            self.schema_override, schema_override)


class Process(Composite, metaclass=abc.ABCMeta):
    """Process parent class

    All :term:`process` classes must inherit from this class.
    """
    defaults: Dict[str, Any] = {}

    def __init__(self, parameters: Optional[dict] = None) -> None:
        super().__init__(parameters)

        self.parameters = self.config
        self.parallel = self.config.pop('_parallel', False)
        if self.config.get('_register'):
            self.register()

    def initial_state(self, config: Optional[dict] = None) -> State:
        """Get initial state in embedded path dictionary

        Every subclass may override this method.

        Arguments:
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

    def register(self, name: Optional[str] = None) -> None:
        process_registry.register(name or self.name, self)

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {self.name: self}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        ports = self.ports()
        return Topology({
            self.name: {
                port: (port,) for port in ports.keys()}})

    def get_schema(self, override: Optional[Schema] = None) -> dict:
        ports = copy.deepcopy(self.ports_schema())
        deep_merge(ports, self.schema_override)
        deep_merge(ports, override)
        return ports

    def merge_overrides(self, override: Schema) -> None:
        deep_merge(self.schema_override, override)

    def ports(self) -> Dict[str, List[str]]:
        ports_schema = self.ports_schema()
        return {
            port: list(states.keys())
            for port, states in ports_schema.items()}

    def local_timestep(self) -> Union[float, int]:
        '''
        Returns the favored timestep for this process. Meant to be
        overridden in subclasses, unless 1.0 is a happy value.
        '''
        return self.parameters.get('time_step', DEFAULT_TIME_STEP)

    def default_state(self) -> State:
        schema = self.ports_schema()
        state = State({})
        for port, states in schema.items():
            for key, value in states.items():
                if '_default' in value:
                    if port not in state:
                        state[port] = {}
                    state[port][key] = value['_default']
        return state

    # The three following methods don't use `self`, but since subclasses
    # might, they need to take `self` as a parameter.

    def is_deriver(self) -> bool:
        return False

    def derivers(self) -> Dict[str, Any]:
        return {}

    def pull_data(self) -> State:
        return State({})

    @abc.abstractmethod
    def ports_schema(self) -> Schema:
        '''
        ports_schema returns a dictionary that declares which states are
        expected by the processes, and how each state will behave.

        state keys can be assigned properties through schema_keys declared
        in Store: '_default', '_updater', _divider', '_value', '_properties',
        '_emit', '_serializer'
        '''
        return Schema({})

    def or_default(self, parameters: Dict[str, Any], key: str) -> Any:
        return parameters.get(key, self.defaults[key])

    def derive_defaults(
            self, original_key: str, derived_key: str,
            f: Callable[[Any], Any]) -> Any:
        source = self.parameters.get(original_key)
        self.parameters[derived_key] = f(source)
        return self.parameters[derived_key]

    @abc.abstractmethod
    def next_update(
            self, timestep: Union[float, int], states: State) -> Update:
        '''
        Find the next update given the current states this process cares
        about. This is the main function a new process would override.
        '''
        return Update({})


class Deriver(Process, metaclass=abc.ABCMeta):
    def is_deriver(self) -> bool:
        return True


def run_update(connection: Connection, process: Process) -> None:
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
        self.process = process
        self.parent, self.child = Pipe()
        self.multiprocess = Multiprocess(
            target=run_update,
            args=(self.child, self.process))
        self.multiprocess.start()

    def update(
            self, interval: Union[float, int], states: State) -> None:
        self.parent.send((interval, states))

    def get(self) -> Tuple[Union[float, int], State]:
        return self.parent.recv()

    def end(self) -> None:
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
            return State({'a_port': {'a': 1}})

        def ports_schema(self) -> Schema:
            return Schema({'a_port': {'a': {'_emit': True}}})

        def next_update(
                self,
                timestep: Union[float, int],
                states: State) -> Update:
            return Update({'a_port': {'a': 1}})

    class BB(Composite):
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
            return Topology({
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
            })

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
        return Schema({
            'A': {
                'a': {'_default': 0},
                'b': {'_default': 0}},
            'B': {
                'a': {'_default': 0},
                'b': {'_default': 0}}})

    def next_update(
            self, timestep: Union[float, int], states: State) -> Update:
        return Update({
            'A': {
                'a': 1,
                'b': states['A']['a']},
            'B': {
                'a': states['A']['b'],
                'b': states['B']['a']}})


class ToyComposite(Composite):
    defaults = {
        'A':  {'name': 'A'},
        'B': {'name': 'B'}}

    def generate_processes(
            self,
            config: Optional[dict] = None) -> Dict[str, ToyProcess]:
        assert config is not None
        return {
            'A': ToyProcess(config['A']),
            'B': ToyProcess(config['B'])}

    def generate_topology(
            self, config: Optional[dict] = None) -> Topology:
        return Topology({
            'A': {
                'A': ('aaa',),
                'B': ('bbb',)},
            'B': {
                'A': ('bbb',),
                'B': ('ccc',)}})


def test_composite_merge() -> None:
    generator = ToyComposite()
    initial_network = generator.generate()

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
    merge_topology = Topology({
        'C': {
            'A': ('aaa',),
            'B': ('bbb',)}})
    generator.merge(
        merge_processes,
        merge_topology)

    config = {'A': {'name': 'D'}, 'B': {'name': 'E'}}
    merged_network = generator.generate(config)

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
        topology=Topology({'b': {
            'A': ('A',),
            'B': ('B',),
        }}))

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
