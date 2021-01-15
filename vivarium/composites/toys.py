from typing import Optional, Dict, Any

from vivarium.core.process import Process, Deriver, Factory


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


class ToyCompartment(Factory):
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
        glucose_required = timestep / \
                           self.parameters['mass_conversion_rate']
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
