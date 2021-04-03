import math
from typing import Optional, Dict, Any, Union

import numpy as np

from vivarium.core.process import (
    Process, Deriver, Composer, Composite, MetaComposer)
from vivarium.core.types import State, Schema, Update, Topology

quark_colors = ['green', 'red', 'blue']
quark_spins = ['up', 'down']
electron_spins = ['-1/2', '1/2']
electron_orbitals = [
    str(orbit) + 's'
    for orbit in range(1, 8)]


class ToyTransport(Process):
    name = 'toy_transport'

    def __init__(
            self,
            initial_parameters: Optional[Dict[str, Any]] = None
    ):
        initial_parameters = initial_parameters or {}
        parameters = {'intake_rate': 2}
        parameters.update(initial_parameters)
        super().__init__(parameters)

    def ports_schema(self):
        ports = {
            'external': ['GLC'],
            'internal': ['GLC']}
        return {
            port_id: {
                key: {
                    '_default': 0.0,
                    '_emit': True}
                for key in keys}
            for port_id, keys in ports.items()}

    def next_update(self, timestep, states):
        update = {}
        intake = timestep * self.parameters['intake_rate']
        if states['external']['GLC'] >= intake:
            update = {
                'external': {'GLC': -2, 'MASS': 1},
                'internal': {'GLC': 2}}

        return update


class ToyDeriveVolume(Deriver):
    name = 'toy_derive_volume'

    def __init__(
            self,
            initial_parameters: Optional[Dict[str, Any]] = None
    ):
        _ = initial_parameters  # ignore initial_parameters
        parameters: Dict[str, Any] = {}
        super().__init__(parameters)

    def ports_schema(self):
        ports = {
            'compartment': ['MASS', 'DENSITY', 'VOLUME']}
        return {
            port_id: {
                key: {
                    '_updater': 'set' if key == 'VOLUME' else
                    'accumulate',
                    '_default': 0.0,
                    '_emit': True}
                for key in keys}
            for port_id, keys in ports.items()}

    def next_update(self, timestep, states):
        volume = states['compartment']['MASS'] /\
                 states['compartment']['DENSITY']
        update = {
            'compartment': {'VOLUME': volume}}

        return update


class ToyDeath(Process):
    name = 'toy_death'

    def __init__(
            self,
            initial_parameters: Optional[Dict[str, Any]] = None
    ):
        initial_parameters = initial_parameters or {}
        self.targets = initial_parameters.get('targets', [])
        super().__init__({})

    def ports_schema(self):
        return {
            'compartment': {
                'VOLUME': {
                    '_default': 0.0,
                    '_emit': True}},
            'global': {
                target: {
                    '_default': None}
                for target in self.targets}}

    def next_update(self, timestep, states):
        volume = states['compartment']['VOLUME']
        update = {}

        if volume > 1.0:
            # kill the cell
            update = {
                'global': {
                    '_delete': [
                        (target,)
                        for target in self.targets]}}

        return update


class ToyCompartment(Composer):
    '''
    a toy compartment for testing

    '''
    def __init__(self, config):
        super().__init__(config)

    def generate_processes(self, config):
        return {
            'metabolism': ToyMetabolism(
                {'mass_conversion_rate': 0.5}),
            'transport': ToyTransport(),
            'death': ToyDeath({'targets': [
                'metabolism',
                'transport']}),
            'external_volume': ToyDeriveVolume(),
            'internal_volume': ToyDeriveVolume()
        }

    def generate_topology(self, config):
        return{
            'metabolism': {
                'pool': 'cytoplasm'},
            'transport': {
                'external': 'periplasm',
                'internal': 'cytoplasm'},
            'death': {
                'global': '',
                'compartment': 'cytoplasm'},
            'external_volume': {
                'compartment': 'periplasm'},
            'internal_volume': {
                'compartment': 'cytoplasm'}}


class ToyMetabolism(Process):
    name = 'toy_metabolism'

    def __init__(
            self,
            initial_parameters: Optional[Dict[str, Any]] = None
    ):
        initial_parameters = initial_parameters or {}
        parameters = {'mass_conversion_rate': 1}
        parameters.update(initial_parameters)
        super().__init__(parameters)

    def ports_schema(self):
        ports = {
            'pool': ['GLC', 'MASS']}
        return {
            port_id: {
                key: {
                    '_default': 0.0,
                    '_emit': True}
                for key in keys}
            for port_id, keys in ports.items()}

    def next_update(self, timestep, states):
        update = {}
        glucose_required = (timestep /
                            self.parameters['mass_conversion_rate'])
        if states['pool']['GLC'] >= glucose_required:
            update = {
                'pool': {
                    'GLC': -2,
                    'MASS': 1}}

        return update


class ToyLinearGrowthDeathProcess(Process):

    name = 'toy_linear_growth_death'

    GROWTH_RATE = 1.0
    THRESHOLD = 6.0

    def __init__(
            self,
            initial_parameters: Optional[Dict[str, Any]] = None
    ):
        initial_parameters = initial_parameters or {}
        self.targets = initial_parameters.get('targets')
        super().__init__(initial_parameters)

    def ports_schema(self):
        return {
            'global': {
                'mass': {
                    '_default': 1.0,
                    '_emit': True}},
            'targets': {
                target: {
                    '_default': None}
                for target in self.targets}}

    def next_update(self, timestep, states):
        mass = states['global']['mass']
        mass_grown = (
            ToyLinearGrowthDeathProcess.GROWTH_RATE * timestep)
        update = {
            'global': {'mass': mass_grown},
        }
        if mass > ToyLinearGrowthDeathProcess.THRESHOLD:
            update['global'] = {
                '_delete': [(target,) for target in self.targets]}

        return update


class Proton(Process):
    name = 'proton'
    defaults = {
        'time_step': 1.0,
        'radius': 0.0}

    def ports_schema(self):
        return {
            'radius': {
                '_updater': 'set',
                '_default': self.parameters['radius']},
            'quarks': {
                '_divider': 'split_dict',
                '*': {
                    'color': {
                        '_updater': 'set',
                        '_default': quark_colors[0]},
                    'spin': {
                        '_updater': 'set',
                        '_default': quark_spins[0]}}},
            'electrons': {
                '*': {
                    'orbital': {
                        '_updater': 'set',
                        '_default': electron_orbitals[0]},
                    'spin': {
                        '_default': electron_spins[0]}}}}

    def next_update(self, timestep, states):
        update = {}

        collapse = np.random.uniform()
        if collapse < states['radius'] * timestep:
            update['radius'] = collapse
            update['quarks'] = {}

            for name in states['quarks'].keys():
                update['quarks'][name] = {
                    'color': np.random.choice(quark_colors),
                    'spin': np.random.choice(quark_spins)}

            update['electrons'] = {}
            orbitals = electron_orbitals.copy()
            for name in states['electrons'].keys():
                np.random.shuffle(orbitals)
                update['electrons'][name] = {
                    'orbital': orbitals.pop()}

        return update


class Electron(Process):
    name = 'electron'
    defaults = {
        'time_step': 1.0,
        'spin': electron_spins[0]}

    def ports_schema(self):
        return {
            'spin': {
                '_updater': 'set',
                '_default': self.parameters['spin']},
            'proton': {
                'radius': {
                    '_default': 0.0}}}

    def next_update(self, timestep, states):
        update = {}

        if np.random.uniform() < states['proton']['radius']:
            update['spin'] = np.random.choice(electron_spins)

        return update


class Sine(Process):
    name = 'sine'
    defaults = {
        'initial_phase': 0.0}

    def ports_schema(self):
        return {
            'frequency': {
                '_default': 440.0},
            'amplitude': {
                '_default': 1.0},
            'phase': {
                '_default': self.parameters['initial_phase']},
            'signal': {
                '_default': 0.0,
                '_updater': 'set'}}

    def next_update(self, timestep, states):
        phase_shift = timestep * states['frequency'] % 1.0
        signal = states['amplitude'] * math.sin(
            2 * math.pi * (states['phase'] + phase_shift))

        return {
            'phase': phase_shift,
            'signal': signal}


class ExchangeA(Process):
    """ Exchange A

    A minimal exchange process that moves molecules 'A' between internal
    and external ports
    """
    name = 'exchange_a'
    defaults = {
        'uptake_rate': 0.0,
        'secrete_rate': 0.0}

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.uptake_rate = self.parameters['uptake_rate']
        self.secrete_rate = self.parameters['secrete_rate']

    def ports_schema(self):
        return {
            'internal': {
                'A': {
                    '_default': 0.0,
                    '_emit': True}},
            'external': {
                'A': {
                    '_default': 0.0,
                    '_emit': True}}}

    def next_update(self, timestep, states):
        a_in = states['internal']['A']
        a_out = states['external']['A']
        delta_a_in = a_out * self.uptake_rate - a_in * self.secrete_rate
        return {
            'internal': {'A': delta_a_in},
            'external': {'A': -delta_a_in}}


class Po(Process):
    name = 'po'

    def initial_state(self, config=None):
        return {
            'A': {
                'a1': -1,
                'a2': -2,
                'a3': -3},
            'B': {
                'b1': -4,
                'b2': -5}}

    def ports_schema(self):
        return {
            'A': {
                'a1': {'_default': 0},
                'a2': {'_default': 0},
                'a3': {'_default': 0}},
            'B': {
                'b1': {'_default': 0},
                'b2': {'_default': 0}}}

    def next_update(self, timestep, states):
        return {
            'A': {
                'a1': 1,
                'a2': 1,
                'a3': 1},
            'B': {
                'b1': -1,
                'b2': -1}}


class Qo(Process):
    name = 'qo'

    def initial_state(self, config=None):
        return {
            'D': {
                'd1': 1,
                'd2': 2,
                'd3': 3},
            'E': {
                'e1': 4,
                'e2': 5}}

    def ports_schema(self):
        return {
            'D': {
                'd1': {'_default': 0},
                'd2': {'_default': 0},
                'd3': {'_default': 0}},
            'E': {
                'e1': {'_default': 0},
                'e2': {'_default': 0}}}

    def next_update(self, timestep, states):
        return {
            'D': {
                'd1': 10,
                'd2': 10,
                'd3': 10},
            'E': {
                'e1': -10,
                'e2': -10}}


class PoQo(Composer):
    def generate_processes(self, config=None):
        return {
            'po': Po(config),
            'qo': Qo(config),
        }

    def generate_topology(self, config=None):
        return {
            'po': {
                'A': {
                    '_path': 'aaa',
                    'a2': 'x',
                    'a3': '<ccc>a3'},
                'B': 'bbb',
            },
            'qo': {
                'D': {
                    '_path': '',
                    'd1': 'aaa>d1',
                    'd2': 'aaa>d2',
                    'd3': 'ccc>d3'},
                'E': {
                    '_path': '',
                    'e1': 'aaa>x',
                    'e2': 'bbb>e2'}
            },
        }


def test_composite_initial_state_complex() -> None:

    outer_path = ('universe', 'agent')
    pq = PoQo({})
    pq_composite = pq.generate(path=outer_path)
    pq_initial = pq_composite.initial_state()

    expected_initial = {
        'universe': {
            'agent': {
                'aaa': {'a1': -1, 'd1': 1, 'd2': 2, 'x': 4},
                'bbb': {'b1': -4, 'b2': -5, 'e2': 5},
                'ccc': {'a3': -3, 'd3': 3}
            }}}

    assert pq_initial == expected_initial


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
            'a3_store': {
                'a3': AA({})}}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'a1': {
                'a_port': ('a1_store',)},
            'a2': {
                'a_port': {
                    'a': ('a1_store', 'b')}},
            'a3_store': {
                'a3': {
                    'a_port': ('a3_1_store',)}}}


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


class ToyComposer(Composer):
    defaults = {
        'A':  {'name': 'A'},
        'B': {'name': 'B'}}

    def generate_processes(
            self,
            config: Optional[dict]
    ) -> Dict[str, ToyProcess]:
        assert config is not None
        config = config or self.defaults
        a = ToyProcess(config['A'])
        b = ToyProcess(config['B'])
        return {
            a.name: a,
            b.name: b}

    def generate_topology(
            self,
            config: Optional[dict] = None
    ) -> Topology:
        config = config or self.defaults
        a_name = config['A']['name']
        b_name = config['B']['name']
        return {
            a_name: {
                'A': ('aaa',),
                'B': ('bbb',)},
            b_name: {
                'A': ('bbb',),
                'B': ('ccc',)}}


def test_override() -> None:
    config = {
        '_schema': {
            'a3_store': {
                'a3': {
                    'a_port': {
                        'a': {
                            '_default': 2}}}}}}
    bb_composer = BB(config)
    bb_composite = bb_composer.generate()
    default_state = bb_composite.default_state()

    expected_default_state = {
        'a3_store': {
            'a3_1_store': {
                'a': 2}}}

    assert default_state == expected_default_state


def test_composite_initial_state() -> None:
    """
    test that initial state in composite merges individual processes'
    initial states
    """

    bb_composer = BB({})
    bb_composite = bb_composer.generate()

    composer_initial_state = bb_composer.initial_state()
    composite_initial_state = bb_composite.initial_state()

    expected_initial_state = {
        'a3_store': {
            'a3_1_store': {
                'a': 1}},
        'a1_store': {
            'a': 1,
            'b': 1}}

    assert composite_initial_state == composer_initial_state
    assert composite_initial_state == expected_initial_state


def test_composite_parameters() -> None:
    """
    test that initial state in composite merges individual processes'
    initial states
    """

    bb_composer = BB({})
    bb_composite = bb_composer.generate()
    composer_parameters = bb_composer.get_parameters()
    composite_parameters = bb_composite.get_parameters()
    expected_parameters = {
        'a1': {'time_step': 1.0},
        'a2': {'time_step': 1.0},
        'a3_store': {
            'a3': {'time_step': 1.0}}}
    assert composite_parameters == composer_parameters
    assert composite_parameters == expected_parameters


def test_composite_merge() -> None:
    composer = ToyComposer()
    composite = composer.generate()

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
    assert composite['topology'] == expected_initial_topology

    for key in ('A', 'B'):
        assert key in composite['processes']
        assert isinstance(composite['processes'][key], ToyProcess)

    # merge
    merge_processes: Dict[str, Process] = {
        'C': ToyProcess({'name': 'C'})}
    merge_topology: Topology = {
        'C': {
            'A': ('aaa',),
            'B': ('bbb',)}}
    composite.merge(
        processes=merge_processes,
        topology=merge_topology)

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
    assert composite['topology'] == expected_merged_topology

    for key in ('A', 'B', 'C'):
        assert key in composite['processes']
        assert isinstance(composite['processes'][key], ToyProcess)


def test_get_composite() -> None:
    a = ToyProcess({'name': 'a'})

    composite = Composite(dict(
        processes={'b': ToyProcess()},
        topology={'b': {'A': ('A',), 'B': ('B',)}}
    ))
    composite.merge(
        processes={a.name: a},
        topology=a.generate_topology())

    expected_topology = {
        'a': {
            'A': ('A',),
            'B': ('B',)},
        'b': {
            'A': ('A',),
            'B': ('B',)}}

    assert composite['topology'] == expected_topology


def test_aggregate_composer() -> None:
    config1 = {'name': 'one'}
    aggregate = MetaComposer(
        composers=[ToyComposer(config1)])
    composite1 = aggregate.generate()

    # add composers (list)
    config2 = {
        'name': 'two',
        'A':  {'name': 'AA'},
        'B': {'name': 'BB'},
    }
    aggregate.add_composers(
        composers=[ToyComposer(config2)],
        config={})
    composite2 = aggregate.generate()

    # add composer (single)
    config3 = {
        'name': 'three',
        'A':  {'name': 'AAA'},
        'B': {'name': 'BBB'},
    }
    aggregate.add_composer(
        composer=ToyComposer(config3),
        config={})
    composite3 = aggregate.generate()

    assert all(
        item in composite2['processes'].keys()
        for item in composite1['processes'].keys())
    assert all(
        item in composite3['processes'].keys()
        for item in composite2['processes'].keys())
    assert all(
        item in composite2['topology'].keys()
        for item in composite1['topology'].keys())
    assert all(
        item in composite3['topology'].keys()
        for item in composite2['topology'].keys())
    assert len(composite1['processes']) < len(composite2['processes'])
    assert len(composite2['processes']) < len(composite3['processes'])

if __name__ == '__main__':
    test_composite_initial_state()  # pragma: no cover
    test_composite_parameters()  # pragma: no cover
    test_composite_merge()  # pragma: no cover
    test_get_composite()  # pragma: no cover
    test_aggregate_composer()  # pragma: no cover
    test_override()  # pragma: no cover
    test_composite_initial_state_complex()  # pragma: no cover
