import numpy as np

from vivarium.core.engine import Engine
from vivarium.core.composer import Composer
from vivarium.library.units import units
from vivarium.processes.divide_condition import DivideCondition
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.growth_rate import GrowthRate


TIMESTEP = 10


class AgentDivision(Composer):
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

    def generate_processes(self, config) -> dict:
        division_config = {
            'daughter_path': config['daughter_path'],
            'agent_id': config['agent_id'],
            'composer': self}
        time_step_config = {'time_step': config['time_step']}
        return {
            'growth': GrowthRate({**config['growth'], **time_step_config}),
            'divide_condition': DivideCondition(config['divide_condition']),
            'division': MetaDivision(division_config)}

    def generate_topology(self, config) -> dict:
        boundary_path = config['boundary_path']
        agents_path = config['agents_path']
        return {
            'growth': {
                'variables': boundary_path,
                'rates': ('rates',)},
            'divide_condition': {
                'variable': boundary_path + ('volume',),
                'divide': boundary_path + ('divide',)},
            'division': {
                'global': boundary_path,
                'agents': agents_path}}


def test_hierarchy_update() -> None:
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

    # make a txtl composite, embedded under an 'agents' store
    txtl_composer = AgentDivision({})
    txtl_composite1 = txtl_composer.generate(
        {'agent_id': agent_id},
        path=('agents', agent_id))

    # check the agents before division
    for n in ['processes', 'steps', 'topology', 'flow']:
        assert '0' in txtl_composite1[n]['agents'], f'agent 0 not in {n}'

    # make the experiment
    hierarchy_experiment1 = Engine(
        composite=txtl_composite1,
        initial_state=initial_state,
        emit_step=100.0)

    # run the experiment long enough to divide
    hierarchy_experiment1.update(2000)

    # check that the agents updated after division
    for n in ['processes', 'steps', 'topology']:  # TODO: add flow
        assert '0' not in txtl_composite1[n]['agents'], \
            f'agent 0 not removed from {n}'
        assert '00' in txtl_composite1[n]['agents'], \
            f'agent 00 not added to {n}'
        assert '01' in txtl_composite1[n]['agents'], \
            f'agent 01 not added to {n}'


if __name__ == '__main__':
    test_hierarchy_update()
