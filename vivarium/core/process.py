"""
==========================================
Factory, Process, and Composite Classes
==========================================
"""

import copy
import numpy as np
import abc
from typing import Any, Dict, Optional

from bson.objectid import ObjectId
from multiprocessing import Pipe
from multiprocessing import Process as Multiprocess
from pint.errors import UndefinedUnitError

from vivarium.library.topology import inverse_topology
from vivarium.library.units import Quantity
from vivarium.core.registry import process_registry, serializer_registry
from vivarium.library.dict_utils import deep_merge, deep_merge_check

DEFAULT_TIME_STEP = 1.0


def serialize_value(value):
    if isinstance(value, dict):
        return serialize_dictionary(value)
    elif isinstance(value, list):
        return serialize_list(value)
    elif isinstance(value, tuple):
        return serialize_list(list(value))
    elif isinstance(value, np.ndarray):
        return serializer_registry.access('numpy').serialize(value)
    elif isinstance(value, Quantity):
        return serializer_registry.access('units').serialize(value)
    elif callable(value):
        return serializer_registry.access('function').serialize(value)
    elif isinstance(value, Process):
        return serialize_dictionary(serializer_registry.access('process').serialize(value))
    elif isinstance(value, Factory):
        return serialize_dictionary(
            serializer_registry.access('factory').serialize(value))
    elif isinstance(value, (np.integer, np.floating)):
        return serializer_registry.access('numpy_scalar').serialize(value)
    elif isinstance(value, ObjectId):
        return str(value)
    else:
        return value


def deserialize_value(value):
    if isinstance(value, dict):
        return deserialize_dictionary(value)
    elif isinstance(value, list):
        return deserialize_list(value)
    elif isinstance(value, str):
        try:
            return serializer_registry.access('units').deserialize(value)
        except UndefinedUnitError:
            return value
    else:
        return value


def serialize_list(lst):
    serialized = []
    for value in lst:
        serialized.append(serialize_value(value))
    return serialized


def serialize_dictionary(d):
    serialized = {}
    for key, value in d.items():
        if not isinstance(key, str):
            key = str(key)
        serialized[key] = serialize_value(value)
    return serialized


def deserialize_list(lst):
    deserialized = []
    for value in lst:
        deserialized.append(deserialize_value(value))
    return deserialized


def deserialize_dictionary(d):
    deserialized = {}
    for key, value in d.items():
        deserialized[key] = deserialize_value(value)
    return deserialized


def assoc_in(d, path, value):
    if path:
        return dict(d, **{path[0]: assoc_in(d.get(path[0], {}), path[1:], value)})
    else:
        return value


def override_schemas(overrides, processes):
    for key, override in overrides.items():
        process = processes[key]
        if isinstance(process, Process):
            process.merge_overrides(override)
        else:
            override_schemas(override, process)


def generate_derivers(processes, topology):
    deriver_processes = {}
    deriver_topology = {}
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
                    for target, source in config.get('port_mapping', {}).items():
                        path = subtopology[source]
                        deriver_topology[deriver_key][target] = path
        else:
            subderivers = generate_derivers(node, subtopology)
            deriver_processes[process_key] = subderivers['processes']
            deriver_topology[process_key] = subderivers['topology']
    return {
        'processes': deriver_processes,
        'topology': deriver_topology}


def get_composite_initial_state(processes, topology):
    initial_state = {}
    for path, node in processes.items():
        if isinstance(node, dict):
            for key in node.keys():
                initial_state[key] = get_composite_initial_state(node, topology[path])
        elif isinstance(node, Process):
            process_topology = topology[path]
            process_state = node.initial_state()
            process_path = tuple()
            state = inverse_topology(process_path, process_state, process_topology)
            initial_state = deep_merge(initial_state, state)

    return initial_state


class Factory(metaclass=abc.ABCMeta):
    """Factory parent class

    All :term:`factory` classes must inherit from this class.
    """
    defaults: Dict[str, Any] = {}

    def __init__(self, config=None):
        if config is None:
            config = {}
        if 'name' in config:
            self.name = config['name']
        elif not hasattr(self, 'name'):
            self.name = self.__class__.__name__

        self.config = copy.deepcopy(self.defaults)
        self.config = deep_merge(self.config, config)

    @abc.abstractmethod
    def generate_processes(self, config):
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
    def generate_topology(self, config):
        """Generate topology dictionary

        Every subclass must override this method.

        Arguments:
            config (dict): A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            dict: Subclass implementations must return a :term:`topology`
            dictionary.
        """
        return {}

    def generate(self, config=None, path=tuple()):
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

        return {
            'processes': assoc_in({}, path, processes),
            'topology': assoc_in({}, path, topology)}


class Composite(Factory):
    """Composite parent class

    All :term:`composite` classes must inherit from this class.
    """

    def __init__(self, config=None):
        super().__init__(config)

        self.merge_processes = self.config.pop('_processes', {})
        self.merge_topology = self.config.pop('_topology', {})
        self.schema_override = self.config.pop('_schema', {})

    def initial_state(self, config=None):
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
        topology = network['topology']
        initial_state = get_composite_initial_state(processes, topology)
        return initial_state

    def generate_processes(self, config):
        return {}

    def generate_topology(self, config):
        return {}

    def generate(self, config=None, path=tuple()):
        network = super().generate(config=config)
        processes = network['processes']
        topology = network['topology']

        # add merged processes
        # TODO - this assumes all merge_processes are already initialized.
        # TODO - make option to initialize new processes here
        processes = deep_merge(processes, self.merge_processes)
        topology = deep_merge(topology, self.merge_topology)

        override_schemas(self.schema_override, processes)

        return {
            'processes': assoc_in({}, path, processes),
            'topology': assoc_in({}, path, topology)}

    def get_parameters(self):
        network = self.generate({})
        processes = network['processes']
        return {
            process_id: process.parameters
            for process_id, process in processes.items()}

    def merge(
            self,
            processes: Optional[Dict[str, Any]] = None,
            topology: Optional[Dict[str, Any]] = None,
            schema_override: Optional[Dict[str, Any]] = None):
        processes = processes or {}
        topology = topology or {}
        schema_override = schema_override or {}

        for name, process in processes.items():
            assert isinstance(process, Process)

        self.merge_processes = deep_merge_check(self.merge_processes, processes)
        self.merge_topology = deep_merge(self.merge_topology, topology)
        self.schema_override = deep_merge(self.schema_override, schema_override)


class Process(Composite, metaclass=abc.ABCMeta):
    """Process parent class

    All :term:`process` classes must inherit from this class.
    """
    defaults: Dict[str, Any] = {}

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.parameters = self.config
        self.parallel = self.config.pop('_parallel', False)
        if self.config.get('_register'):
            self.register()

    def initial_state(self, config=None):
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
        raise Exception('{} does not include an "initial_state" function'.format(self.name))

    def register(self, name=None):
        process_registry.register(name or self.name, self)

    def generate_processes(self, config):
        return {self.name: self}

    def generate_topology(self, config):
        ports = self.ports()
        return {
            self.name: {
                port: (port,) for port in ports.keys()}}

    def get_schema(self, override=None):
        ports = copy.deepcopy(self.ports_schema())
        deep_merge(ports, self.schema_override)
        deep_merge(ports, override)
        return ports

    def merge_overrides(self, override):
        deep_merge(self.schema_override, override)

    def ports(self):
        ports_schema = self.ports_schema()
        return {
            port: list(states.keys())
            for port, states in ports_schema.items()}

    def local_timestep(self):
        '''
        Returns the favored timestep for this process.
        Meant to be overridden in subclasses, unless 1.0 is a happy value.
        '''
        return self.parameters.get('time_step', DEFAULT_TIME_STEP)

    def default_state(self):
        schema = self.ports_schema()
        state = {}
        for port, states in schema.items():
            for key, value in states.items():
                if '_default' in value:
                    if port not in state:
                        state[port] = {}
                    state[port][key] = value['_default']
        return state

    def is_deriver(self):
        return False

    def derivers(self):
        return {}

    def pull_data(self):
        return {}

    @abc.abstractmethod
    def ports_schema(self):
        '''
        ports_schema returns a dictionary that declares which states are expected by the processes,
        and how each state will behave.

        state keys can be assigned properties through schema_keys declared in Store:
            '_default'
            '_updater'
            '_divider'
            '_value'
            '_properties'
            '_emit'
            '_serializer'
        '''
        return {}

    def or_default(self, parameters, key):
        return parameters.get(key, self.defaults[key])

    def derive_defaults(self, original_key, derived_key, f):
        source = self.parameters.get(original_key)
        self.parameters[derived_key] = f(source)
        return self.parameters[derived_key]

    @abc.abstractmethod
    def next_update(self, timestep, states):
        '''
        Find the next update given the current states this process cares about.
        This is the main function a new process would override.
        '''
        return {}


class Deriver(Process, metaclass=abc.ABCMeta):
    def is_deriver(self):
        return True


def run_update(connection, process):
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


class ParallelProcess(object):
    def __init__(self, process):
        self.process = process
        self.parent, self.child = Pipe()
        self.multiprocess = Multiprocess(
            target=run_update,
            args=(self.child, self.process))
        self.multiprocess.start()

    def update(self, interval, states):
        self.parent.send((interval, states))

    def get(self):
        return self.parent.recv()

    def end(self):
        self.parent.send((-1, None))
        self.multiprocess.join()


def test_composite_initial_state():
    """
    test that initial state in composite merges individual processes' initial states
    """
    class AA(Process):
        name = 'AA'

        def initial_state(self, config=None):
            return {'a_port': {'a': 1}}

        def ports_schema(self):
            return {'a_port': {'a': {'_emit': True}}}

        def next_update(self, timestep, states):
            return {'a_port': {'a': 1}}

    class BB(Composite):
        name = 'BB'

        def generate_processes(self, config):
            return {
                'a1': AA({}),
                'a2': AA({}),
                'a3': {
                    'a3_store': AA({})}
            }

        def generate_topology(self, config):
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

    def ports_schema(self):
        return {
            'A': {
                'a': {'_default': 0},
                'b': {'_default': 0}},
            'B': {
                'a': {'_default': 0},
                'b': {'_default': 0}}}

    def next_update(self, timestep, states):
        return {
            'A': {
                'a': 1,
                'b': states['A']['a']},
            'B': {
                'a': states['A']['b'],
                'b': states['B']['a']}}


class ToyComposite(Composite):
    defaults = {
        'A':  {'name': 'A'},
        'B': {'name': 'B'}}

    def generate_processes(self, config=None):
        return {
            'A': ToyProcess(config['A']),
            'B': ToyProcess(config['B'])}

    def generate_topology(self, config=None):
        return {
            'A': {
                'A': ('aaa',),
                'B': ('bbb',)},
            'B': {
                'A': ('bbb',),
                'B': ('ccc',)}}


def test_composite_merge():

    generator = ToyComposite()
    initial_network = generator.generate()

    # merge
    merge_processes = {
        'C': ToyProcess({'name': 'C'})}
    merge_topology = {
        'C': {
            'A': ('aaa',),
            'B': ('bbb',)}}
    generator.merge(
        merge_processes,
        merge_topology)

    config = {'A': {'name': 'D'}, 'B': {'name': 'E'}}
    merged_network = generator.generate(config)


def test_get_composite():
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
    # test_composite_initial_state()
    # test_composite_merge()

    test_get_composite()
