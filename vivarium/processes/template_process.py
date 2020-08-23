from __future__ import absolute_import, division, print_function

import os

from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process_in_experiment,
    plot_simulation_output,
    PROCESS_OUT_DIR,
)


# a global NAME is used for the output directory and for the process name
NAME = 'template'


class Template(Process):
    '''
    This mock process provides a basic template that can be used for a new process
    '''

    # give the process a name, so that it can register in the process_repository
    name = NAME

    # declare default parameters as class variables
    defaults = {
        'parameter_1': {}
    }

    def __init__(self, initial_parameters=None):
        if initial_parameters is None:
            initial_parameters = {}

        # get the parameters out of initial_parameters if available, or use defaults
        parameter_1 = self.or_default(
            initial_parameters, 'parameter_1')

        parameters = {'parameter_1': parameter_1}
        super(Template, self).__init__(parameters)

    def ports_schema(self):
        '''
        ports_schema returns a dictionary that declares how each state will behave.
        Each key can be assigned settings for the schema_keys declared in Store:
            '_default'
            '_updater'
            '_divider'
            '_value'
            '_properties'
            '_emit'
            '_serializer'
        '''

        return {
            'internal': {
                'A': {
                    '_default': 1.0,
                    '_updater': 'accumulate',
                    '_emit': True,
                }
            },
            'external': {
                'B': {
                    '_default': 1.0,
                    '_updater': 'accumulate',
                    '_emit': True,
                }
            },
        }

    def derivers(self):
        '''
        declare which derivers are needed for this process
        '''
        return {}

    def next_update(self, timestep, states):
        # get the states
        A_state = states['internal']['A']
        B_state = states['external']['B']

        # calculate output for the states
        A_update = 0.5 * B_state
        B_update = -0.5 * B_state

        # return an update to the states
        return {
            'internal': {'A': A_update},
            'external': {'B': B_update}}


# functions to configure and run the process
def run_template_process(out_dir='out'):
    # initialize the process by passing initial_parameters
    initial_parameters = {}
    template_process = Template(initial_parameters)

    # run the simulation
    sim_settings = {'total_time': 10}
    output = simulate_process_in_experiment(template_process, sim_settings)

    # plot the simulation output
    plot_settings = {}
    plot_simulation_output(output, plot_settings, out_dir)


# run module is run as the main program with python vivarium/process/template_process.py
if __name__ == '__main__':
    # make an output directory to save plots
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_template_process(out_dir)
