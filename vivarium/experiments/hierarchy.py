import numpy as np
import math

from vivarium.core.process import Process, Deriver
from vivarium.core.composer import Composer
from vivarium.plots.topology import plot_topology
from vivarium.library.units import units
from vivarium.core.registry import process_registry
from vivarium.processes.divide_condition import DivideCondition
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.growth_rate import GrowthRate
from vivarium.core.composition import composite_in_experiment


TIMESTEP = 10

PI = math.pi


def length_from_volume(volume, width):
    """
    get cell length from volume, using the following equation for capsule volume, with V=volume, r=radius,
    a=length of cylinder without rounded caps, l=total length:

    V = (4/3)*PI*r^3 + PI*r^2*a
    l = a + 2*r
    """
    radius = width / 2
    cylinder_length = (volume - (4/3) * PI * radius**3) / (PI * radius**2)
    total_length = cylinder_length + 2 * radius
    return total_length


class Tl(Process):

    defaults = {
        'ktrl': 5e-4,
        'kdeg': 5e-5}

    def ports_schema(self):
        return {
            'mRNA': {
                'C': {
                    '_default': 100 * units.mg / units.mL,
                    '_divider': 'split',
                    '_emit': True}},
            'Protein': {
                'X': {
                    '_default': 200 * units.mg / units.mL,
                    '_divider': 'split',
                    '_emit': True}}}

    def next_update(self, timestep, states):
        C = states['mRNA']['C']
        X = states['Protein']['X']
        dX = (self.parameters['ktrl'] * C - self.parameters['kdeg'] * X) * timestep
        return {
            'Protein': {
                'X': dX}}


class StochasticTx(Process):
    defaults = {'ktsc': 1e0, 'kdeg': 1e-3}

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
            'DNA': {'G': 1.0},
            'mRNA': {'C': 1.0}}

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


class LengthFromVolume(Deriver):
    defaults = {'width': 1.}  # um
    def ports_schema(self):
        return {
            'global': {
                'volume': {'_default': 1 * units.fL},
                'length': {'_default': 2., '_updater': 'set'},
            }}
    def next_update(self, timestep, states):
        volume = states['global']['volume']
        length = length_from_volume(volume.magnitude, self.parameters['width'])
        return {
            'global': {
                'length': length}}


class TxTlDivision(Composer):
    defaults = {
        'time_step': TIMESTEP,
        'stochastic_Tx': {},
        'Tl': {},
        'concs': {
            'molecular_weights': {
                'C': 1e8 * units.g / units.mol}},
        'growth': {
            'time_step': 1,
            'default_growth_rate': 0.0005,
            'default_growth_noise': 0.001,
            'variables': ['volume']},
        'agent_id': np.random.randint(0, 100),
        'divide_condition': {
            'threshold': 2.5 * units.fL},
        'agents_path': ('..', '..', 'agents',),
        'boundary_path': ('boundary',),
        'daughter_path': tuple(),
        '_schema': {
            'concs': {
                'input': {'C': {'_divider': 'binomial'}},
                'output': {'C': {'_divider': 'set'}},
            }}}

    def generate_processes(self, config):
        counts_to_concentration = process_registry.access('counts_to_concentration')
        division_config = dict(
            daughter_path=config['daughter_path'],
            agent_id=config['agent_id'],
            composer=self)
        time_step_config = {'time_step': config['time_step']}
        return {
            'stochastic_Tx': StochasticTx({**config['stochastic_Tx'], **time_step_config}),
            'Tl': Tl({**config['Tl'], **time_step_config}),
            'concs': counts_to_concentration(config['concs']),
            'growth': GrowthRate({**config['growth'], **time_step_config}),
            'divide_condition': DivideCondition(config['divide_condition']),
            'shape': LengthFromVolume(),
            'division': MetaDivision(division_config)}

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        agents_path = config['agents_path']
        return {
            'stochastic_Tx': {
                'DNA': ('DNA',),
                'mRNA': ('RNA_counts',)},
            'Tl': {
                'mRNA': ('RNA',),
                'Protein': ('Protein',)},
            'concs': {
                'global': boundary_path,
                'input': ('RNA_counts',),
                'output': ('RNA',)},
            'growth': {
                'variables': boundary_path,
                'rates': ('rates',)},
            'divide_condition': {
                'variable': boundary_path + ('volume',),
                'divide': boundary_path + ('divide',)},
            'shape': {
                'global': boundary_path,
            },
            'division': {
                'global': boundary_path,
                'agents': agents_path}}


def test_hierarchy_update():
    # configure hierarchy
    agent_id = '0'

    # initial state
    # initial state
    initial_state = {
        'agents': {
            agent_id: {
                'boundary': {'volume': 1.2 * units.fL},
                'DNA': {'G': 1},
                'RNA': {'C': 5 * units.mg / units.mL},
                'Protein': {'X': 50 * units.mg / units.mL}}}}

    # experiment settings
    exp_settings = {
        'experiment_id': 'hierarchy_experiment',
        'initial_state': initial_state,
        'emit_step': 100.0}

    # make a txtl composite, embedded under an 'agents' store
    txtl_composer = TxTlDivision({})
    txtl_composite1 = txtl_composer.generate({'agent_id': agent_id}, path=('agents', agent_id))

    # make the experiment
    hierarchy_experiment1 = composite_in_experiment(
        composite=txtl_composite1,
        settings=exp_settings,
        initial_state=initial_state)

    # run the experiment long enough to divide
    hierarchy_experiment1.update(2000)

    # plot the topology
    fig = plot_topology(
        txtl_composite1,
        out_dir='out',
        filename='hierarchy_topology.pdf',
        )


if __name__ == '__main__':
    test_hierarchy_update()
