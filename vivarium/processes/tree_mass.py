'''
=========
Tree Mass
=========
'''

import os

from scipy import constants

from vivarium.core.experiment import pp
from vivarium.core.process import Deriver
from vivarium.library.units import units
from vivarium.core.composition import (
    process_in_experiment,
    PROCESS_OUT_DIR,
)


NAME = 'mass_deriver'
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

    name = NAME
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

    def initial_state(self, config=None):
        return {
            'global': {
                'initial_mass': self.parameters['initial_mass'],
                'mass': self.parameters['initial_mass'],
            }
        }

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


def test_tree_mass():

    mass_1 = 1.0 * units.g / units.mol
    mass_2 = 2.0 * units.g / units.mol

    # declare schema override to get mw properties
    parameters = {
        'initial_mass': 0 * units.g,  # in grams
        '_schema': {
            'A': {
                '1': {
                    '_emit': True,
                    '_properties': {'mw': mass_1}},
                '2': {
                    '_emit': True,
                    '_properties': {'mw': mass_2}},
            },
            'B': {
                '1': {
                    '_emit': True,
                    '_properties': {'mw': mass_1}},
                '2': {
                    '_emit': True,
                    '_properties': {'mw': mass_2}},
            },
        }
    }
    mass_process = TreeMass(parameters)

    # declare initial state
    state = {
        'A': {
            '1': 2.0 * AVOGADRO.magnitude,
            '2': 0.0,
        },
        'B': {
            '1': 0.0,
            '2': 1.0 * AVOGADRO.magnitude,
        },
        'global': {
            'initial_mass': 0.0,
            'mass': 0.0,
        }
    }

    # make the experiment with initial state
    settings = {'initial_state': state}
    experiment = process_in_experiment(mass_process, settings)

    # run experiment and get output
    experiment.update(1)
    output = experiment.emitter.get_data_deserialized()
    experiment.end()

    assert output[0.0]['global']['mass'] == 4 * units.g
    return output


def _run_tree_mass():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    os.makedirs(out_dir, exist_ok=True)
    output = test_tree_mass()
    pp(output)


if __name__ == '__main__':
    _run_tree_mass()
