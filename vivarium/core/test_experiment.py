import logging as log
from multiprocessing import Pool

from vivarium.composites.toys import Proton, Electron, Sine
from vivarium.core.experiment import Experiment, pf, pp, MultiInvoke
from vivarium.core.tree import hierarchy_depth, Store
from vivarium.core.process import Process, Composite
from vivarium.library.units import units


def make_proton(parallel=False):
    processes = {
        'proton': Proton({'_parallel': parallel}),
        'electrons': {
            'a': {
                'electron': Electron({'_parallel': parallel})},
            'b': {
                'electron': Electron()}}}

    spin_path = ('internal', 'spin')
    radius_path = ('structure', 'radius')

    topology = {
        'proton': {
            'radius': radius_path,
            'quarks': ('internal', 'quarks'),
            'electrons': {
                '_path': ('electrons',),
                '*': {
                    'orbital': ('shell', 'orbital'),
                    'spin': spin_path}}},
        'electrons': {
            'a': {
                'electron': {
                    'spin': spin_path,
                    'proton': {
                        '_path': ('..', '..'),
                        'radius': radius_path}}},
            'b': {
                'electron': {
                    'spin': spin_path,
                    'proton': {
                        '_path': ('..', '..'),
                        'radius': radius_path}}}}}

    initial_state = {
        'structure': {
            'radius': 0.7},
        'internal': {
            'quarks': {
                'x': {
                    'color': 'green',
                    'spin': 'up'},
                'y': {
                    'color': 'red',
                    'spin': 'up'},
                'z': {
                    'color': 'blue',
                    'spin': 'down'}}}}

    return {
        'processes': processes,
        'topology': topology,
        'initial_state': initial_state}


def test_recursive_store():
    environment_config = {
        'environment': {
            'temperature': {
                '_default': 0.0,
                '_updater': 'accumulate'},
            'fields': {
                (0, 1): {
                    'enzymeX': {
                        '_default': 0.0,
                        '_updater': 'set'},
                    'enzymeY': {
                        '_default': 0.0,
                        '_updater': 'set'}},
                (0, 2): {
                    'enzymeX': {
                        '_default': 0.0,
                        '_updater': 'set'},
                    'enzymeY': {
                        '_default': 0.0,
                        '_updater': 'set'}}},
            'agents': {
                '1': {
                    'location': {
                        '_default': (0, 0),
                        '_updater': 'set'},
                    'boundary': {
                        'external': {
                            '_default': 0.0,
                            '_updater': 'set'},
                        'internal': {
                            '_default': 0.0,
                            '_updater': 'set'}},
                    'transcripts': {
                        'flhDC': {
                            '_default': 0,
                            '_updater': 'accumulate'},
                        'fliA': {
                            '_default': 0,
                            '_updater': 'accumulate'}},
                    'proteins': {
                        'ribosome': {
                            '_default': 0,
                            '_updater': 'set'},
                        'flagella': {
                            '_default': 0,
                            '_updater': 'accumulate'}}},
                '2': {
                    'location': {
                        '_default': (0, 0),
                        '_updater': 'set'},
                    'boundary': {
                        'external': {
                            '_default': 0.0,
                            '_updater': 'set'},
                        'internal': {
                            '_default': 0.0,
                            '_updater': 'set'}},
                    'transcripts': {
                        'flhDC': {
                            '_default': 0,
                            '_updater': 'accumulate'},
                        'fliA': {
                            '_default': 0,
                            '_updater': 'accumulate'}},
                    'proteins': {
                        'ribosome': {
                            '_default': 0,
                            '_updater': 'set'},
                        'flagella': {
                            '_default': 0,
                            '_updater': 'accumulate'}}}}}}

    state = Store(environment_config)
    state.apply_update({})
    state.state_for(['environment'], ['temperature'])


def test_topology_ports():
    proton = make_proton()

    experiment = Experiment(proton)

    log.debug(pf(experiment.state.get_config(True)))

    experiment.update(10.0)

    log.debug(pf(experiment.state.get_config(True)))
    log.debug(pf(experiment.state.divide_value()))


def test_timescales():
    class Slow(Process):
        name = 'slow'
        defaults = {'timestep': 3.0}

        def __init__(self, config=None):
            super().__init__(config)

        def ports_schema(self):
            return {
                'state': {
                    'base': {
                        '_default': 1.0}}}

        def local_timestep(self):
            return self.parameters['timestep']

        def next_update(self, timestep, states):
            base = states['state']['base']
            next_base = timestep * base * 0.1

            return {
                'state': {'base': next_base}}

    class Fast(Process):
        name = 'fast'
        defaults = {'timestep': 0.3}

        def __init__(self, config=None):
            super().__init__(config)

        def ports_schema(self):
            return {
                'state': {
                    'base': {
                        '_default': 1.0},
                    'motion': {
                        '_default': 0.0}}}

        def local_timestep(self):
            return self.parameters['timestep']

        def next_update(self, timestep, states):
            base = states['state']['base']
            motion = timestep * base * 0.001

            return {
                'state': {'motion': motion}}

    processes = {
        'slow': Slow(),
        'fast': Fast()}

    states = {
        'state': {
            'base': 1.0,
            'motion': 0.0}}

    topology = {
        'slow': {'state': ('state',)},
        'fast': {'state': ('state',)}}

    emitter = {'type': 'null'}
    experiment = Experiment({
        'processes': processes,
        'topology': topology,
        'emitter': emitter,
        'initial_state': states})

    experiment.update(10.0)


def test_2_store_1_port():
    """
    Split one port of a processes into two stores
    """
    class OnePort(Process):
        name = 'one_port'

        def ports_schema(self):
            return {
                'A': {
                    'a': {
                        '_default': 0,
                        '_emit': True},
                    'b': {
                        '_default': 0,
                        '_emit': True}
                }
            }

        def next_update(self, timestep, states):
            return {
                'A': {
                    'a': 1,
                    'b': 2}}

    class SplitPort(Composite):
        """splits OnePort's ports into two stores"""
        name = 'split_port_generator'

        def generate_processes(self, config):
            return {
                'one_port': OnePort({})}

        def generate_topology(self, config):
            return {
                'one_port': {
                    'A': {
                        'a': ('internal', 'a',),
                        'b': ('external', 'a',)
                    }
                }}

    # run experiment
    split_port = SplitPort({})
    network = split_port.generate()
    exp = Experiment({
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'external': {'a': [0, 2, 4]},
        'internal': {'a': [0, 1, 2]},
        'time': [0.0, 1.0, 2.0]}
    assert output == expected_output


def test_multi_port_merge():
    class MultiPort(Process):
        name = 'multi_port'

        def ports_schema(self):
            return {
                'A': {
                    'a': {
                        '_default': 0,
                        '_emit': True}},
                'B': {
                    'a': {
                        '_default': 0,
                        '_emit': True}},
                'C': {
                    'a': {
                        '_default': 0,
                        '_emit': True}}}

        def next_update(self, timestep, states):
            return {
                'A': {'a': 1},
                'B': {'a': 1},
                'C': {'a': 1}}

    class MergePort(Composite):
        """combines both of MultiPort's ports into one store"""
        name = 'multi_port_generator'

        def generate_processes(self, config):
            return {
                'multi_port': MultiPort({})}

        def generate_topology(self, config):
            return {
                'multi_port': {
                    'A': ('aaa',),
                    'B': ('aaa',),
                    'C': ('aaa',)}}

    # run experiment
    merge_port = MergePort({})
    network = merge_port.generate()
    exp = Experiment({
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'aaa': {'a': [0, 3, 6]},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


def test_complex_topology():
    class Po(Process):
        name = 'po'

        def ports_schema(self):
            return {
                'A': {
                    'a': {'_default': 0},
                    'b': {'_default': 0},
                    'c': {'_default': 0}},
                'B': {
                    'd': {'_default': 0},
                    'e': {'_default': 0}}}

        def next_update(self, timestep, states):
            return {
                'A': {
                    'a': states['A']['b'],
                    'b': states['A']['c'],
                    'c': states['B']['d'] + states['B']['e']},
                'B': {
                    'd': states['A']['a'],
                    'e': states['B']['e']}}

    class Qo(Process):
        name = 'qo'

        def ports_schema(self):
            return {
                'D': {
                    'x': {'_default': 0},
                    'y': {'_default': 0},
                    'z': {'_default': 0}},
                'E': {
                    'u': {'_default': 0},
                    'v': {'_default': 0}}}

        def next_update(self, timestep, states):
            return {
                'D': {
                    'x': -1,
                    'y': 12,
                    'z': states['D']['x'] + states['D']['y']},
                'E': {
                    'u': 3,
                    'v': states['E']['u']}}

    class PoQo(Composite):
        def generate_processes(self, config=None):
            p = Po(config)
            q = Qo(config)

            return {
                'po': p,
                'qo': q}

        def generate_topology(self, config=None):
            return {
                'po': {
                    'A': {
                        '_path': ('aaa',),
                        'b': ('o',)},
                    'B': ('bbb',)},
                'qo': {
                    'D': {
                        'x': ('aaa', 'a'),
                        'y': ('aaa', 'o'),
                        'z': ('ddd', 'z')},
                    'E': {
                        'u': ('aaa', 'u'),
                        'v': ('bbb', 'e')}}}

    initial_state = {
        'aaa': {
            'a': 2,
            'c': 5,
            'o': 3,
            'u': 11},
        'bbb': {
            'd': 14,
            'e': 88},
        'ddd': {
            'z': 333}}

    pq = PoQo({})
    pq_config = pq.generate()
    pq_config['initial_state'] = initial_state

    experiment = Experiment(pq_config)

    pp(experiment.state.get_value())
    experiment.update(1)

    state = experiment.state.get_value()
    assert state['aaa']['a'] == initial_state['aaa']['a'] + \
           initial_state['aaa']['o'] - 1
    assert state['aaa']['o'] == initial_state['aaa']['o'] + \
           initial_state['aaa']['c'] + 12
    assert state['aaa']['c'] == initial_state['aaa']['c'] + \
           initial_state['bbb']['d'] + initial_state['bbb']['e']
    assert state['aaa']['u'] == initial_state['aaa']['u'] + \
           3
    assert state['bbb']['d'] == initial_state['bbb']['d'] + \
           initial_state['aaa']['a']
    assert state['bbb']['e'] == initial_state['bbb']['e'] + \
           initial_state['bbb']['e'] + initial_state['aaa']['u']
    assert state['ddd']['z'] == initial_state['ddd']['z'] + \
           initial_state['aaa']['a'] + initial_state['aaa']['o']


def test_multi():
    with Pool(processes=4) as pool:
        multi = MultiInvoke(pool)
        proton = make_proton()
        experiment = Experiment({**proton, 'invoke': multi.invoke})

        log.debug(pf(experiment.state.get_config(True)))

        experiment.update(10.0)

        log.debug(pf(experiment.state.get_config(True)))
        log.debug(pf(experiment.state.divide_value()))


def test_parallel():
    proton = make_proton(parallel=True)
    experiment = Experiment(proton)

    log.debug(pf(experiment.state.get_config(True)))

    experiment.update(10.0)

    log.debug(pf(experiment.state.get_config(True)))
    log.debug(pf(experiment.state.divide_value()))

    experiment.end()


def test_depth():
    nested = {
        'A': {
            'AA': 5,
            'AB': {
                'ABC': 11}},
        'B': {
            'BA': 6}}

    dissected = hierarchy_depth(nested)
    assert len(dissected) == 3
    assert dissected[('A', 'AB', 'ABC')] == 11


def test_sine():
    sine = Sine()
    print(sine.next_update(0.25 / 440.0, {
        'frequency': 440.0,
        'amplitude': 0.1,
        'phase': 1.5}))


def test_units():
    class UnitsMicrometer(Process):
        name = 'units_micrometer'

        def ports_schema(self):
            return {
                'A': {
                    'a': {
                        '_default': 0 * units.um,
                        '_emit': True},
                    'b': {
                        '_default': 'string b',
                        '_emit': True,
                    }
                }
            }

        def next_update(self, timestep, states):
            return {
                'A': {'a': 1 * units.um}}

    class UnitsMillimeter(Process):
        name = 'units_millimeter'

        def ports_schema(self):
            return {
                'A': {
                    'a': {
                        # '_default': 0 * units.mm,
                        '_emit': True}}}

        def next_update(self, timestep, states):
            return {
                'A': {'a': 1 * units.mm}}

    class MultiUnits(Composite):
        name = 'multi_units_generator'

        def generate_processes(self, config):
            return {
                'units_micrometer':
                    UnitsMicrometer({}),
                'units_millimeter':
                    UnitsMillimeter({})}

        def generate_topology(self, config):
            return {
                'units_micrometer': {
                    'A': ('aaa',)},
                'units_millimeter': {
                    'A': ('aaa',)}}

    # run experiment
    multi_unit = MultiUnits({})
    network = multi_unit.generate()
    exp = Experiment({
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(5)
    timeseries = exp.emitter.get_timeseries()
    print('TIMESERIES')
    pp(timeseries)

    data = exp.emitter.get_data()
    print('DATA')
    pp(data)

    data_deserialized = exp.emitter.get_data_deserialized()
    print('DESERIALIZED')
    pp(data_deserialized)

    data_unitless = exp.emitter.get_data_unitless()
    print('UNITLESS')
    pp(data_unitless)


if __name__ == '__main__':
    # test_recursive_store()
    # test_timescales()
    # test_topology_ports()
    # test_multi()
    # test_sine()
    # test_parallel()
    # test_complex_topology()
    # test_multi_port_merge()
    # test_2_store_1_port()

    test_units()
