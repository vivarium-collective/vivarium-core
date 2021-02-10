import math
from typing import Optional, Dict, Any

import numpy as np

from vivarium.core.process import Process, Deriver, Composer

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
                'pool': ('cytoplasm',)},
            'transport': {
                'external': ('periplasm',),
                'internal': ('cytoplasm',)},
            'death': {
                'global': tuple(),
                'compartment': ('cytoplasm',)},
            'external_volume': {
                'compartment': ('periplasm',)},
            'internal_volume': {
                'compartment': ('cytoplasm',)}}


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
                    '_path': ('aaa',),
                    'a2': ('x',),
                    'a3': ('..', 'ccc', 'a3')},
                'B': ('bbb',),
            },
            'qo': {
                'D': {
                    '_path': (),
                    'd1': ('aaa', 'd1'),
                    'd2': ('aaa', 'd2'),
                    'd3': ('ccc', 'd3')},
                'E': {
                    '_path': (),
                    'e1': ('aaa', 'x'),
                    'e2': ('bbb', 'e2')}
            },
        }
