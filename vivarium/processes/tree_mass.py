from __future__ import absolute_import, division, print_function

from scipy import constants

from vivarium.core.process import Deriver
from vivarium.library.units import units

AVOGADRO = constants.N_A * 1 / units.mol


def calculate_mass(value, path, node):
    if 'mw' in node.properties:
        count = node.value
        mw = node.properties['mw']
        mol = count / AVOGADRO
        added_mass = mw * mol
        return value + added_mass
    else:
        return value


class TreeMass(Deriver):
    """
    Derives and sets total mass from individual molecular counts
    that have a mass schema in their stores .

    """

    name = 'mass_deriver'
    defaults = {
        'from_path': ('..', '..'),
        'initial_mass': 0 * units.fg,
    }

    def __init__(self, parameters=None):
        super(TreeMass, self).__init__(parameters)
        self.from_path = self.parameters['from_path']

    def ports_schema(self):
        return {
            'global': {
                'initial_mass': {
                    '_default': self.parameters['initial_mass'],
                    '_updater': 'set',
                    '_divider': 'split',
                },
                'mass': {
                    '_default': self.parameters['initial_mass'],
                    '_emit': True,
                    '_updater': 'set',
                    '_divider': 'split',
                },
            },
        }

    def next_update(self, timestep, states):
        initial_mass = states['global']['initial_mass']

        return {
            'global': {
                'mass': {
                    '_reduce': {
                        'reducer': calculate_mass,
                        'from': self.from_path,
                        'initial': initial_mass}}}}

# register process by invoking upon import
TreeMass()
