'''
=========
Tree Mass
=========
'''

from __future__ import absolute_import, division, print_function

from scipy import constants

from vivarium.core.process import Deriver
from vivarium.library.units import units

AVOGADRO = constants.N_A * 1 / units.mol


def calculate_mass(value, path, node):
    '''Reducer for summing masses in hierarchy

    Arguments:
        value: The value to add mass to.
        path: Unused.
        node: The node whose mass will be added.

    Returns:
        The mass of the node (accounting for the node's molecular
        weight, which should be stored in its ``mw`` property) added to
        ``value``.
    '''
    if 'mw' in node.properties:
        count = node.value
        mw = node.properties['mw']
        mol = count / AVOGADRO
        added_mass = mw * mol
        return value + added_mass
    else:
        return value


class TreeMass(Deriver):

    name = 'mass_deriver'
    defaults = {
        'from_path': ('..', '..'),
        'initial_mass': 0 * units.fg,
    }

    def __init__(self, parameters=None):
        """Derive total mass from molecular counts and weights.

        Arguments:
            parameters (dict): Dictionary of parameters. The following
                keys are required:

                * **from_path** (:py:class:`tuple`): Path to the root of
                  the subtree whose mass will be summed.
        """
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
        '''Return a ``_reduce`` update to store the total mass.

        Store mass in ``('global', 'mass')``.
        '''
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
