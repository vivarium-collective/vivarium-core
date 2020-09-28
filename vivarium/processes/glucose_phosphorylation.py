"""
===================================
Toy Glucose Phosphorylation Process
===================================

This is a toy example referenced in the documentation.
"""

from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process,
)
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium.processes.tree_mass import TreeMass
from vivarium.library.units import units


class GlucosePhosphorylation(Process):

    name = 'glucose_phosphorylation'
    defaults = {
        'k_cat': 2e-3,
        'K_ATP': 5e-2,
        'K_GLC': 4e-2,
    }

    def __init__(self, initial_parameters=None):
        parameters = GlucosePhosphorylation.defaults
        parameters.update(initial_parameters)
        super(GlucosePhosphorylation, self).__init__(
            parameters)

    def next_update(self, timestep, states):
        # Get concentrations from state
        cytoplasm = states['cytoplasm']
        nucleoside_phosphates = states['nucleoside_phosphates']
        hk = cytoplasm['HK']
        glc = cytoplasm['GLC']
        atp = nucleoside_phosphates['ATP']

        # Get kinetic parameters
        k_cat = self.parameters['k_cat']
        k_atp = self.parameters['K_ATP']
        k_glc = self.parameters['K_GLC']

        # Compute reaction rate with michaelis-menten equation
        rate = k_cat * hk * glc * atp / (
            k_glc * k_atp + k_glc * atp + k_atp * glc + glc * atp)

        # Compute concentration changes from rate and timestep
        delta_glc = -rate * timestep
        delta_atp = -rate * timestep
        delta_g6p = rate * timestep
        delta_adp = rate * timestep

        # Compile changes into an update
        update = {
            'cytoplasm': {
                'GLC': delta_glc,
                'G6P': delta_g6p,
                # We exclude HK because it doesn't change
            },
            'nucleoside_phosphates': {
                'ATP': delta_atp,
                'ADP': delta_adp,
            },
        }

        return update

    def ports_schema(self):
        return {
            'cytoplasm': {
                'GLC': {
                    # accumulate means to add the updates
                    '_updater': 'accumulate',
                    '_default': 1.0,
                    '_properties': {
                        'mw': 1.0 * units.g / units.mol,
                    },
                    '_emit': True,
                },
                # accumulate is the default, so we don't need to specify
                # updaters for the rest of the variables
                'G6P': {
                    '_default': 0.0,
                    '_properties': {
                        'mw': 1.0 * units.g / units.mol,
                    },
                    '_emit': True,
                },
                'HK': {
                    '_default': 0.1,
                    '_properties': {
                        'mw': 1.0 * units.g / units.mol,
                    },
                },
            },
            'nucleoside_phosphates': {
                'ATP': {
                    '_default': 2.0,
                    '_emit': True,
                },
                'ADP': {
                    '_default': 0.0,
                    '_emit': True,
                }
            },
            'global': {
            },
        }

    def derivers(self):
        # We instantiate TreeMass to make sure it's added to the deriver
        # registry.
        TreeMass()
        return {
            'my_deriver': {
                'deriver': 'mass_deriver',
                'port_mapping': {
                    'global': 'global',
                },
                'config': {},
            },
        }


if __name__ == '__main__':
    parameters = {
        'k_cat': 1.5,
    }
    my_process = GlucosePhosphorylation(parameters)

    settings = {
        'total_time': 10,
        'timestep': 0.1,
    }
    timeseries = simulate_process(my_process, settings)
    plot_simulation_output(timeseries, {}, './')
