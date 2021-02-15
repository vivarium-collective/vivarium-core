from vivarium.core.process import Deriver
from vivarium.library.units import units
from vivarium.processes.tree_mass import AVOGADRO


class DeriveCounts(Deriver):
    """
    Process for deriving counts from concentrations
    """
    name = 'counts_deriver'
    defaults = {
        'concentration_keys': [],
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def initial_state(self, config=None):
        mass = 1000 * units.fg
        density = 1100 * units.g / units.L
        volume = mass / density
        return {
            'global': {
                'volume': volume.to('fL'),
            }}

    def ports_schema(self):
        initial_state = self.initial_state()
        return {
            'global': {
                'volume': {
                    '_default': initial_state['global']['volume'].to('fL')},
            },
            'counts': {
                molecule: {
                    '_default': 0,
                    '_divider': 'split'}
                for molecule in self.parameters['concentration_keys']},
            'concentrations': {
                molecule: {
                    '_default': 0.0}
                for molecule in self.parameters['concentration_keys']}}

    def next_update(self, timestep, states):
        volume = states['global']['volume']
        mmol_to_counts = (AVOGADRO * volume).to('L/mmol').magnitude
        concentrations = states['concentrations']  # assumes mmol/L

        counts = {}
        for molecule, concentration in concentrations.items():
            counts[molecule] = {
                '_value': int(concentration * mmol_to_counts),
                '_updater': 'set'}

        return {
            'counts': counts}
