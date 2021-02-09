"""
Toy model of stochastic transcription, composed with deterministic translation
"""
import os
import numpy as np

from vivarium.core.process import Process, Composite
from vivarium.core.composition import (
    process_in_experiment,
    compartment_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.core.registry import process_registry
from vivarium.plots.simulation_output import plot_simulation_output



class StochasticTSC(Process):
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
        G = states['DNA']['G']
        C = states['mRNA']['C']

        array_state = np.array([G, C])

        # Calculate propensities
        propensities = [self.ktsc * array_state[0], self.kdeg * array_state[1]]
        prop_sum = sum(propensities)

        # The wait time is distributed exponentially
        self.calculated_timestep = np.random.exponential(scale=prop_sum)
        return self.calculated_timestep

    def next_reaction(self, X):

        propensities = [self.ktsc * X[0], self.kdeg * X[1]]
        prop_sum = sum(propensities)

        # Choose the next reaction
        r_rxn = np.random.random()
        for i in range(len(propensities)):
            if r_rxn < propensities[i] / prop_sum:
                # This means propensity i fires
                break
        X += self.stoichiometry[i]

        return X

    def next_update(self, timestep, states):

        if self.time_left is not None:
            if timestep >= self.time_left:
                event = self.event
                self.event = None
                self.time_left = None
                return event
            else:
                self.time_left -= timestep
                return {}
        else:
            # retrieve the state values, put them in array
            G = states['DNA']['G']
            C = states['mRNA']['C']
            array_state = np.array([G, C])

            # calculate the next reaction
            new_state = self.next_reaction(array_state)

            # get delta mRNA
            C1 = new_state[1]
            dC = C1 - C

            update = {
                'mRNA': {
                    'C': dC}}

            if self.calculated_timestep > timestep:
                # didn't get all of our time, need to store the event for later
                self.time_left = self.calculated_timestep - timestep
                self.event = update
                return {}
            else:
                # return an update
                return {
                    'mRNA': {
                        'C': dC}}


class TRL(Process):

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
        C = states['mRNA']['C']
        X = states['Protein']['X']
        dX = (self.parameters['ktrl'] * C - self.parameters['kdeg'] * X) * timestep
        return {
            'Protein': {
                'X': dX}}


class TrlConcentration(TRL):
    """rescale mRNA"""

    def next_update(self, timestep, states):
        states['mRNA']['C'] = states['mRNA']['C'].magnitude * 1e5
        return super().next_update(timestep, states)


class StochasticTscTrl(Composite):
    defaults = {
        'stochastic_TSC': {'time_step': 10},
        'TRL': {'time_step': 10},
    }

    def generate_processes(self, config):
        concentrations_deriver = process_registry.access('concentrations_deriver')
        return {
            'stochastic_TSC': StochasticTSC(config['stochastic_TSC']),
            'TRL': TrlConcentration(config['TRL']),
            'concs': concentrations_deriver({'concentration_keys': ['C']})
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

    # run it and retrieve the data that was emitted to the simulation log
    for era in range(total_time):
        gillespie_experiment.update(1)

    gillespie_data = gillespie_experiment.emitter.get_timeseries()
    return gillespie_data

def test_gillespie_composite(total_time=10000):
    stochastic_tsc_trl = StochasticTscTrl()

    # make the experiment
    exp_settings = {
        'experiment_id': 'stochastic_tsc_trl'}
    stoch_experiment = compartment_in_experiment(
        stochastic_tsc_trl,
        exp_settings)

    # simulate and retrieve the data from emitter
    stoch_experiment.update(total_time)
    data = stoch_experiment.emitter.get_timeseries()

    return data


def main():
    out_dir = os.path.join(PROCESS_OUT_DIR, 'toy_gillespie')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    process_output = test_gillespie_process()
    composite_output = test_gillespie_composite()

    # plot the simulation output
    plot_settings = {}
    plot_simulation_output(process_output, plot_settings, out_dir, filename='process')
    plot_simulation_output(composite_output, plot_settings, out_dir, filename='composite')


if __name__ == '__main__':
    main()
