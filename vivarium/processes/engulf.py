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
            'trigger': {
                '_default': [],
                '_emit': True,
            },
            'internal': {},
            'agents': {}}

    def next_update(self, timestep, states):
        if states['trigger']:
            neighbor_ids = states['trigger']
            return {
                'agents': {
                    '_move': [{
                        'source': (id,),
                        'target': (self.agent_id, 'internal',)  # TODO -- 'internal' is not necessarily the true path.
                    } for id in neighbor_ids]
                }
            }
        else:
            return {}


# test
class ToyAgent(Generator):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'engulf': {}}

    def generate_processes(self, config):
        engulf_config = config['engulf']
        engulf_config['agent_id'] = config['agent_id']
        return {
            'exchange': ExchangeA(config['exchange']),
            'engulf': Engulf(engulf_config)}

    def generate_topology(self, config):
        agents_path = ('..', '..', 'agents')
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'engulf': {
                'trigger': ('trigger',),
                'internal': ('subcompartments',),
                'agents': agents_path}}


def test_disintegrate():
    agent_1_id = '1'
    agent_2_id = '2'

    # initial state
    initial_state = {
        'agents': {
            agent_1_id: {
                'external': {'A': 1},
                'trigger': []},
            agent_2_id: {
                'external': {'A': 1},
                'trigger': []}}}

    # timeline triggers engulf for agent_1
    time_engulf = 5
    time_total = 10
    timeline = [
        (0, {('agents', agent_1_id, 'trigger'): []}),
        (time_engulf, {('agents', agent_1_id, 'trigger'): [agent_2_id]}),
        (time_total, {})]

    # declare the hierarchy
    hierarchy = {
        'processes': [
            {
                'type': TimelineProcess,
                'config': {'timeline': timeline},
                'topology': {
                    'global': ('global',),
                    'agents': ('agents',)
                }
            }
        ],
        'agents': {
            'generators': [
                {
                    'name': agent_1_id,
                    'type': ToyAgent,
                    'config': {'agent_id': agent_1_id}
                },
                {
                    'name': agent_2_id,
                    'type': ToyAgent,
                    'config': {'agent_id': agent_2_id}
                },
            ]
        }
    }

    # configure experiment
    experiment = compartment_hierarchy_experiment(
        hierarchy=hierarchy,
        initial_state=initial_state)

    # run simulation
    experiment.update(time_total)
    output = experiment.emitter.get_data()
    experiment.end()  # end required for parallel processes

    # assert len(output['agents']['1']['dead']) == time_dead + 1
    # assert len(output['time']) == time_total + 1

    import ipdb;
    ipdb.set_trace()

    return output

def run_disintegrate():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output = test_disintegrate()
    pp(output)


if __name__ == '__main__':
    run_disintegrate()
