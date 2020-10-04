"""
===============
Engulf Process
===============
"""

import os
import uuid
import logging as log

from vivarium.core.experiment import pp
from vivarium.core.process import (
    Deriver,
    Generator,
)
from vivarium.library.units import units
from vivarium.core.composition import (
    compartment_hierarchy_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium.processes.exchange_a import ExchangeA
from vivarium.processes.timeline import TimelineProcess


NAME = 'engulf'


class Engulf(Deriver):
    """ Engulf Process

    remove a compartment when the state under the 'trigger' port is set to True.
    """
    name = NAME
    defaults = {}

    def __init__(self, parameters=None):
        super(Engulf, self).__init__(parameters)
        self.agent_id = self.parameters['agent_id']

    def ports_schema(self):
        return {
            'engulf': {
                '_default': False,
                '_emit': True},
            'agents': {}}

    def next_update(self, timestep, states):
        if states['engulf']:
            # TODO - engulf what neighbor?
            neighbor_id = states['engulf']['neighbor_id']

            return {
                'agents': {
                    'node': [(neighbor_id)],
                    'path': daughter_updates}
            }
        else:
            return {}


# test
class ToyAgent(Generator):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'death': {}
    }

    def generate_processes(self, config):
        death_config = config['death']
        death_config['agent_id'] = config['agent_id']
        return {
            'exchange': ExchangeA(config['exchange']),
            'engulf': Engulf(death_config)}

    def generate_topology(self, config):
        agents_path = ('..', '..', 'agents')
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'engulf': {
                # set the trigger to be the 'dead' state
                'trigger': ('engulf',),
                'agents': agents_path}}


def test_disintegrate():
    agent_1_id = '1'
    agent_2_id = '2'

    # initial state
    initial_state = {
        'agents': {
            agent_1_id: {
                'external': {'A': 1},
                'trigger': False},
            agent_2_id: {
                'external': {'A': 1},
                'trigger': False}
        }
    }

    # timeline triggers engulf for agent_1
    time_engulf = 5
    time_total = 10
    timeline = [
        (0, {('agents', agent_1_id, 'engulf'): False}),
        (time_engulf, {
            ('agents', agent_1_id, 'engulf'): {
                'agent_id': agent_2_id
            }}),
        (time_total, {})]

    # declare the hierarchy
    hierarchy = {
        'processes': [
            {
                'type': TimelineProcess,
                'config': {
                    'timeline': timeline},
                'topology': {
                    'global': ('global',),
                    'agents': ('agents',)
                },
            }
        ],
        'agents': {
            'generators': [
                {
                    'name': agent_1_id,
                    'type': ToyAgent,
                    'config': {'agent_id': agent_1_id},
                    # 'topology': {},
                },
                {
                    'name': agent_2_id,
                    'type': ToyAgent,
                    'config': {'agent_id': agent_2_id},
                    # 'topology': {},
                },
            ]
        }
    }

    # configure experiment
    experiment = compartment_hierarchy_experiment(
        hierarchy=hierarchy,
        initial_state=initial_state,
    )

    import ipdb; ipdb.set_trace()

    # run simulation
    experiment.update(total_time)
    output = experiment.emitter.get_data()
    experiment.end()  # end required for parallel processes

    # assert len(output['agents']['1']['dead']) == time_dead + 1
    # assert len(output['time']) == time_total + 1

    return output

def run_disintegrate():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output = test_disintegrate()
    pp(output)


if __name__ == '__main__':
    run_disintegrate()
