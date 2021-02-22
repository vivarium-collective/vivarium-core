"""
==================
GrowthRate Process
==================
"""

import os
import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import (
    PROCESS_OUT_DIR,
    process_in_experiment,
)
from vivarium.plots.simulation_output import plot_simulation_output

NAME = 'growth_rate'


class GrowthRate(Process):
    """ A Vivarium process that models exponential growth of biomass """

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
                    '_default': 1.0,
                    '_divider': 'split',
                    '_emit': True,
                } for variable in self.parameters['variables']
            },
            'rates': {
                'growth_rate': {
                    variable: {
                        '_default': self.parameters[
                            'default_growth_rate'],
                    } for variable in self.parameters['variables']
                },
                'growth_noise': {
                    variable: {
                        '_default': self.parameters[
                            'default_growth_noise'],
                    } for variable in self.parameters['variables']
                },
            }
        }

    def next_update(self, timestep, states):
        variables = states['variables']
        growth_rate = states['rates']['growth_rate']
        growth_noise = states['rates']['growth_noise']

        variable_update = {
            variable: value * (
                    np.exp(growth_rate[variable] +
                           np.random.normal(0, growth_noise[variable]))
                    * timestep) - value
            for variable, value in variables.items()}
        return {'variables': variable_update}


def test_growth_rate(total_time=1350):
    initial_mass = 100
    growth_rate = 0.0005
    config = {
        'variables': ['mass'],
        'default_growth_rate': growth_rate}

    growth_rate_process = GrowthRate(config)
    initial_state = {'variables': {'mass': initial_mass}}
    experiment = process_in_experiment(
        growth_rate_process,
        initial_state=initial_state)
    experiment.update(total_time)
    output = experiment.emitter.get_timeseries()

    # asserts
    final_mass = output['variables']['mass'][-1]
    expected_mass = initial_mass * np.exp(growth_rate * total_time)
    decimal_precision = 7
    assert abs(expected_mass - final_mass) < \
           1.5 * 10 ** (-decimal_precision)

    return output


def main():
    """run test and plot"""
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = test_growth_rate()
    plot_settings = {}
    plot_simulation_output(data, plot_settings, out_dir)


if __name__ == '__main__':
    main()
