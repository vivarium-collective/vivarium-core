import numpy as np

from vivarium.core.process import Process
from vivarium.core.composer import Composer
from vivarium.plots.topology import plot_topology
from vivarium.library.units import units
from vivarium.processes.divide_condition import DivideCondition
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.growth_rate import GrowthRate
from vivarium.core.composition import composite_in_experiment


TIMESTEP = 10


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


class TlDivision(Composer):
    defaults = {
        'time_step': TIMESTEP,
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
    }

    def generate_processes(self, config):
        division_config = dict(
            daughter_path=config['daughter_path'],
            agent_id=config['agent_id'],
            composer=self)
        time_step_config = {'time_step': config['time_step']}
        return {
            'Tl': Tl({**config['Tl'], **time_step_config}),
            'growth': GrowthRate({**config['growth'], **time_step_config}),
            'divide_condition': DivideCondition(config['divide_condition']),
            'division': MetaDivision(division_config)}

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        agents_path = config['agents_path']
        return {
            'Tl': {
                'mRNA': ('RNA',),
                'Protein': ('Protein',)},
            'growth': {
                'variables': boundary_path,
                'rates': ('rates',)},
            'divide_condition': {
                'variable': boundary_path + ('volume',),
                'divide': boundary_path + ('divide',)},
            'division': {
                'global': boundary_path,
                'agents': agents_path}}


def test_hierarchy_update():
    # configure hierarchy
    agent_id = '0'

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
    txtl_composer = TlDivision({})
    txtl_composite1 = txtl_composer.generate(
        {'agent_id': agent_id},
        path=('agents', agent_id))

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
