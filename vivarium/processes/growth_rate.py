import os
import numpy as np
from typing import Any, Dict, Optional

# import the vivarium process class
from vivarium.core.process import Process

from vivarium.core.composition import (
    PROCESS_OUT_DIR,
    process_in_experiment,
)
from vivarium.plots.simulation_output import plot_simulation_output

NAME = 'growth_rate'


class GrowthRate(Process):
    ''' A Vivarium process that models exponential growth of biomass '''

    name = NAME
    defaults = {
        'default_growth_rate': 0.0005,
        'default_growth_noise': 0.0,
        'variables': ['mass']
    }

    def ports_schema(self):
        return {
            'variables': {
                variable: {
                    '_default': 1000.0,
                    '_emit': True,
                } for variable in self.parameters['variables']
            },
            'rates': {
                'growth_rate': {
                    variable: {
                        '_default': self.parameters['default_growth_rate'],
                    } for variable in self.parameters['variables']
                },
                'growth_noise': {
                    variable: {
                        '_default': self.parameters['default_growth_noise'],
                    } for variable in self.parameters['variables']
                },
            }
        }

    def next_update(self, timestep, states):
        variables = states['variables']
        growth_rate = states['rates']['growth_rate']
        growth_noise = states['rates']['growth_noise']

        variable_update = {
            variable: value * (np.exp(growth_rate[variable] +
                       np.random.normal(0, growth_noise[variable])
                       ) * timestep) - value
            for variable, value in variables.items()}

        return {'variables': variable_update}


def test_growth_rate(total_time=1350):
    growth_rate = GrowthRate({'variables': ['mass']})
    initial_state = {'variables': {'mass': 100}}
    experiment = process_in_experiment(growth_rate, initial_state=initial_state)
    experiment.update(total_time)
    output = experiment.emitter.get_timeseries()
    assert output['variables']['mass'][-1] > output['variables']['mass'][0]
    return output


def main(out_dir):
    data = test_growth_rate()
    plot_settings = {}
    plot_simulation_output(data, plot_settings, out_dir)



if __name__ == '__main__':
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    main(out_dir)
