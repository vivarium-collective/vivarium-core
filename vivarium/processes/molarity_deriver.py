from typing import Dict, Any
from vivarium.core.process import Deriver
from vivarium.library.units import units
from vivarium.processes.tree_mass import AVOGADRO


class CountsToMolar(Deriver):
    """
    Process for deriving molar concentrations from counts
    """

    name = 'counts_to_molar'
    defaults: Dict[str, Any] = {
        'keys': [],
    }

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
                    '_default': initial_state[
                        'global']['volume'].to('fL')}},
            'counts': {
                concentration: {
                    '_divider': 'split'}
                for concentration in self.parameters['keys']},
            'concentrations': {
                concentration: {
                    '_divider': 'set',
                    '_updater': 'set'}
                for concentration in self.parameters['keys']}}

    def next_update(self, timestep, states):
        volume = states['global']['volume']
        mmol_to_counts = (AVOGADRO * volume).to('L/mmol').magnitude
        counts = states['counts']

        # concentration update
        concentrations = {}
        if volume != 0:
            for molecule, count in counts.items():
                concentrations[molecule] = count / mmol_to_counts
            return {
                'concentrations': concentrations}
        print('volume is 0!')
        return {}


class MolarToCounts(Deriver):
    """
    Process for deriving counts from molar concentration
    """
    name = 'molar_to_counts'
    defaults: Dict[str, Any] = {
        'keys': [],
    }

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
                for molecule in self.parameters['keys']},
            'concentrations': {
                molecule: {
                    '_default': 0.0}
                for molecule in self.parameters['keys']}}

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

def test_derivers():
    pass


if __name__ == '__main__':
    test_derivers()
