from vivarium.core.process import Deriver
from vivarium.library.units import units
from vivarium.processes.tree_mass import AVOGADRO

# def get_default_global_state():
#     mass = 1000 * units.fg
#     density = 1100 * units.g / units.L
#     volume = mass / density
#     mmol_to_counts = (AVOGADRO * volume)
#     return {
#         'global': {
#             'volume': volume.to('fL'),
#             'mmol_to_counts': mmol_to_counts.to('L/mmol')
#         }}


class DeriveConcentrations(Deriver):
    """
    Process for deriving concentrations from counts
    """

    name = 'concentrations_deriver'
    defaults = {
        'concentration_keys': [],
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def initial_state(self, config=None):
        mass = 1000 * units.fg
        density = 1100 * units.g / units.L
        volume = mass / density
        mmol_to_counts = (AVOGADRO * volume)
        return {
            'global': {
                'volume': volume.to('fL'),
                'mmol_to_counts': mmol_to_counts.to('L/mmol')
            }}

    def ports_schema(self):
        initial_state = self.initial_state()
        return {
            'global': {
                'volume': {
                    '_default': initial_state['global']['volume'].to('fL')},
                'mmol_to_counts': {
                    '_default': initial_state['global']['mmol_to_counts'].to('L/mmol')}},
            'counts': {
                concentration: {
                    '_divider': 'split'}
                for concentration in self.parameters['concentration_keys']},
            'concentrations': {
                concentration: {
                    '_divider': 'set',
                    '_updater': 'set'}
                for concentration in self.parameters['concentration_keys']}}

    def next_update(self, timestep, states):

        # states
        mmol_to_counts = states['global']['mmol_to_counts']
        counts = states['counts']

        # concentration update
        concentrations = {}
        if mmol_to_counts != 0:
            for molecule, count in counts.items():
                concentrations[molecule] = count / mmol_to_counts

            for molecule, concentration in concentrations.items():
                assert concentration >= 0, 'derived {} concentration < 0'.format(molecule)

            return {
                'concentrations': concentrations}
        else:
            print('mmol_to_counts is 0!')
            return {}
