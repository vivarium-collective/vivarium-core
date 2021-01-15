import numpy as np
from typing import Any, Dict, Optional

# import the vivarium process class
from vivarium.core.process import Process
from vivarium.library.units import units

from vivarium.core.composition import (
    simulate_process_in_experiment,
)


class ExponentialGrowth(Process):
    ''' A Vivarium process that models exponential growth of biomass '''

    name = 'exponential_growth'
    defaults = {
        'growth_rate': 0.0005,
        'growth_noise': 1e-4,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        ''' Constructor accepts parameters, which modify defaults. '''
        super().__init__(parameters)

    def ports_schema(self):
        ''' Ports declare expected inputs/outputs and their schema.

        The ports structure is an embedded dictionary. The keys of the
        dictionary declare the port names, and map to the require state
        structure. Each variable can be declared with a schema, which is
        modular and can be modified to do many different operations on
        the state.

        Here, a "globals" port has a single variable -- "mass" -- with a
        declared schema of updaters, dividers, emit, etc.
        '''

        return {
            'globals': {
                'mass': {
                    '_default': 1000.0 * units.fg,
                    '_updater': 'set',
                    '_divider': 'split',
                    '_emit': True,
                }
            }
        }

    def next_update(self, timestep, states):
        ''' defines an update to the states.

        The update function takes the current state and timestep,
        applies the mathematical model, and returns an update.
        '''

        # 1) Retrieve the states at the start of the timestep.
        mass = states['globals']['mass']

        # 2) Apply the mechanism for the timestep's duration.
        # The exponential function of growth rate is applied to "mass"
        new_mass = mass * (
                np.exp(self.parameters['growth_rate'] +
                       np.random.normal(
                           0, self.parameters['growth_noise'])
                       ) * timestep)

        # 3) Update to variables is returned through the ports
        return {
            'globals': {
                'mass': new_mass
            }
        }


if __name__ == '__main__':
    # initialize the process
    growth_process = ExponentialGrowth({})

    # run a simulation with helper function called
    # simulate_process_in_experiment
    settings = {
        'total_time': 5000,
        'initial_state': {
            'globals': {
                'mass': 1000 * units.fg,
            }
        },
        #     'progress_bar': False,
    }
    grow_output = simulate_process_in_experiment(
        growth_process, settings)
