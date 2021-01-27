"""
Stochastic transcription process
"""
import os
import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import (
    process_in_experiment,
    PROCESS_OUT_DIR,
)
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

    def gillespie(self, X, dT):
        '''
        * X: initial condition
        * dT: how long to simulate
        '''

        t = 0.0
        while t < dT + self.time_remaining:
            # Calculate propensities
            propensities = [self.ktsc * X[0], self.kdeg * X[1]]
            prop_sum = sum(propensities)

            # The wait time is distributed exponentially
            wait_time = np.random.exponential(scale=prop_sum)

            # Reached the end of the simulation interval?
            if wait_time + t >= dT:
                self.time_remaining = dT - t  # save the unaccounted sim time
                break

            t += wait_time

            # Choose the next reaction
            r_rxn = np.random.random()
            for i in range(len(propensities)):
                if r_rxn < propensities[i] / prop_sum:
                    # This means propensity i fires
                    break
            X += self.stoichiometry[i]

        return X

    def next_update(self, timestep, states):

        # retrieve the state values
        G = states['DNA']['G']
        C = states['mRNA']['C']

        # apply the mechanism
        new_state = self.gillespie(
            np.array([G, C]),
            timestep)

        # get delta mRNA
        C1 = new_state[1]
        dC = C1 - C

        # return an update
        return {
            'mRNA': {
                'C': dC}}


def test_gillespie_process():
    # construct TscTrl
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

    #
    # import ipdb; ipdb.set_trace()

    return gillespie_data

def main():
    out_dir = os.path.join(PROCESS_OUT_DIR, 'gillespie')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output = test_gillespie_process()

    # plot the simulation output
    plot_settings = {}
    plot_simulation_output(output, plot_settings, out_dir)


# run module with python vivarium/process/template_process.py
if __name__ == '__main__':
    main()
