""" Toy Stochastic Transcription Process
Toy model of Gillespie algorithm-based  transcription,
and a composite with deterministic translation.

Note: This Process is primarily for testing multi-timestepping.
variables and parameters are hard-coded. Do not use this as a
general stochastic transcription.
"""
import os
import numpy as np

from vivarium.core.process import Process, Composer
from vivarium.core.composition import (
    process_in_experiment,
    composite_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.core.registry import process_registry
from vivarium.plots.simulation_output import plot_simulation_output



class StochasticTSC(Process):
    """stochastic toy transcription"""
    defaults = {
        'ktsc': 5e0,
        'kdeg': 1e-1,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.ktsc = self.parameters['ktsc']
        self.kdeg = self.parameters['kdeg']
        self.stoichiometry = np.array([[0, 1], [0, -1]])
        self.time_left = None
        self.event = None

        # initialize the next timestep
        initial_state = self.initial_state()
        self.calculate_timestep(initial_state)

    def initial_state(self, config=None):
        return {
            'DNA': {
                'G': 1.0
            },
            'mRNA': {
                'C': 1.0
            }
        }

    def ports_schema(self):
        return {
            'DNA': {
                'G': {
                    '_default': 1.0,
                    '_emit': True}},
            'mRNA': {
                'C': {
                    '_default': 1.0,
                    '_emit': True}}}

    def calculate_timestep(self, states):
        # retrieve the state values
        g = states['DNA']['G']
        c = states['mRNA']['C']

        array_state = np.array([g, c])

        # Calculate propensities
        propensities = [
            self.ktsc * array_state[0], self.kdeg * array_state[1]]
        prop_sum = sum(propensities)

        # The wait time is distributed exponentially
        self.calculated_timestep = np.random.exponential(scale=prop_sum)
        return self.calculated_timestep

    def next_reaction(self, x):
        """get the next reaction and return a new state"""

        propensities = [self.ktsc * x[0], self.kdeg * x[1]]
        prop_sum = sum(propensities)

        # Choose the next reaction
        r_rxn = np.random.uniform()
        i = 0
        for i, _ in enumerate(propensities):
            if r_rxn < propensities[i] / prop_sum:
                # This means propensity i fires
                break
        x += self.stoichiometry[i]
        return x

    def next_update(self, timestep, states):

        if self.time_left is not None:
            if timestep >= self.time_left:
                event = self.event
                self.event = None
                self.time_left = None
                return event

            self.time_left -= timestep
            return {}

        # retrieve the state values, put them in array
        g = states['DNA']['G']
        c = states['mRNA']['C']
        array_state = np.array([g, c])

        # calculate the next reaction
        new_state = self.next_reaction(array_state)

        # get delta mRNA
        c1 = new_state[1]
        d_c = c1 - c

        update = {
            'mRNA': {
                'C': d_c}}

        if self.calculated_timestep > timestep:
            # didn't get all of our time, store the event for later
            self.time_left = self.calculated_timestep - timestep
            self.event = update
            return {}

        # return an update
        return {
            'mRNA': {
                'C': d_c}}


class TRL(Process):
    """deterministic toy translation"""

    defaults = {
        'ktrl': 1e-2,
        'kdeg': 1e-4,
        }

    def ports_schema(self):
        return {
            'mRNA': {
                'C': {
                    '_default': 1.0,
                    '_emit': True}},
            'Protein': {
                'X': {
                    '_default': 1.0,
                    '_emit': True}}}

    def next_update(self, timestep, states):
        c = states['mRNA']['C']
        x = states['Protein']['X']
        d_x = (
            self.parameters['ktrl'] * c -
            self.parameters['kdeg'] * x) * timestep
        return {
            'Protein': {
                'X': d_x}}


class TrlConcentration(TRL):
    """rescale mRNA"""

    def next_update(self, timestep, states):
        states['mRNA']['C'] = states['mRNA']['C'] * 1e5
        return super().next_update(timestep, states)


class StochasticTscTrl(Composer):
    """
    composite toy model with stochastic transcription,
    deterministic translation.
    """
    defaults = {
        'stochastic_TSC': {'time_step': 10},
        'TRL': {'time_step': 10},
    }

    def generate_processes(self, config):
        counts_to_molar = process_registry.access(
            'counts_to_molar')
        return {
            'stochastic_TSC': StochasticTSC(config['stochastic_TSC']),
            'TRL': TrlConcentration(config['TRL']),
            'concs': counts_to_molar({'keys': ['C']})
        }

    def generate_topology(self, config):
        return {
            'stochastic_TSC': {
                'DNA': ('DNA',),
                'mRNA': ('mRNA_counts',)
            },
            'TRL': {
                'mRNA': ('mRNA',),
                'Protein': ('Protein',)
            },
            'concs': {
                'counts': ('mRNA_counts',),
                'concentrations': ('mRNA',)}
        }



def test_gillespie_process(total_time=1000):
    gillespie_process = StochasticTSC()

    # make the experiment
    exp_settings = {
        'display_info': False,
        'experiment_id': 'TscTrl'}
    gillespie_experiment = process_in_experiment(
        gillespie_process,
        exp_settings)

    # run the experiment in increments
    for _ in range(total_time):
        gillespie_experiment.update(1)

    gillespie_data = gillespie_experiment.emitter.get_timeseries()
    return gillespie_data

def test_gillespie_composite(total_time=10000):
    stochastic_tsc_trl = StochasticTscTrl().generate()

    # make the experiment
    exp_settings = {
        'experiment_id': 'stochastic_tsc_trl'}
    stoch_experiment = composite_in_experiment(
        stochastic_tsc_trl,
        exp_settings)

    # simulate and retrieve the data from emitter
    stoch_experiment.update(total_time)
    data = stoch_experiment.emitter.get_timeseries()

    return data


def main():
    """run the tests and plot"""
    out_dir = os.path.join(PROCESS_OUT_DIR, 'toy_gillespie')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    process_output = test_gillespie_process()
    composite_output = test_gillespie_composite()

    # plot the simulation output
    plot_settings = {}
    plot_simulation_output(
        process_output, plot_settings, out_dir, filename='process')
    plot_simulation_output(
        composite_output, plot_settings, out_dir, filename='composite')


if __name__ == '__main__':
    main()
