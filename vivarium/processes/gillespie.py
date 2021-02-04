"""
Stochastic transcription process
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
        self.time_remaining = 0.0

        # TODO - initialize with next timestep?


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

    def next_timestep(self, X):

        # Calculate propensities
        propensities = [self.ktsc * X[0], self.kdeg * X[1]]
        prop_sum = sum(propensities)

        # The wait time is distributed exponentially
        wait_time = np.random.exponential(scale=prop_sum)

        return wait_time

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

        local_timestep = self.local_timestep()
        time_left = local_timestep - timestep
        if time_left > 1e-6:
            import ipdb; ipdb.set_trace()
            return {}

        # retrieve the state values
        G = states['DNA']['G']
        C = states['mRNA']['C']

        array_state = np.array([G, C])

        new_state = self.next_reaction(array_state)

        # get delta mRNA
        C1 = new_state[1]
        dC = C1 - C

        next_ts = self.next_timestep(array_state)
        self.set_timestep(next_ts)

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



def test_gillespie_process():
    gillespie_process = StochasticTSC()

    # make the experiment
    exp_settings = {
        'experiment_id': 'TscTrl'}
    gillespie_experiment = process_in_experiment(
        gillespie_process,
        exp_settings)

    # run it and retrieve the data that was emitted to the simulation log
    gillespie_experiment.update(1000)
    gillespie_data = gillespie_experiment.emitter.get_timeseries()

    return gillespie_data

def test_gillespie_composite():
    total_time = 10000

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

    # import ipdb; ipdb.set_trace()

    return data


def main():
    out_dir = os.path.join(PROCESS_OUT_DIR, 'gillespie')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # output = test_gillespie_process()
    output = test_gillespie_composite()

    # # plot the simulation output
    # plot_settings = {}
    # plot_simulation_output(output, plot_settings, out_dir)


# run module with python vivarium/process/template_process.py
if __name__ == '__main__':
    main()
