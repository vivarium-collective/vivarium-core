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
    compose_experiment,
    GENERATORS_KEY,
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
    defaults = {
        'inner_path': ('inner',)
    }

    def __init__(self, parameters=None):
        super(Engulf, self).__init__(parameters)
        self.agent_id = self.parameters['agent_id']
        self.inner_path = self.parameters['inner_path']

    def ports_schema(self):
        ''' trigger list includes ids of things to engulf '''
        return {
            'trigger': {
                '_default': [],
            },
            'inner': {
                '*': {}
            },
            'outer': {
                '*': {}
            }
        }

    def next_update(self, timestep, states):
        if states['trigger']:
            neighbor_ids = states['trigger']
            # move neighbors from outer to inner, reset trigger
            return {
                'trigger': {
                    '_updater': 'set',
                    '_value': []},
                'outer': {
                    '_move': [{
                        'source': (id,),
                        'target': (self.agent_id,) + self.inner_path
                    } for id in neighbor_ids]
                }
            }
        else:
            return {}


# test
class ToyAgent(Generator):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'outer_path': ('..', '..', 'agents'),
        'inner_path': ('subcompartments',),
    }

    def generate_processes(self, config):
        agent_id = config['agent_id']
        outer_path = config['outer_path']
        inner_path = config['inner_path']
        engulf_config = dict(
            outer_path=outer_path,
            inner_path=inner_path,
            agent_id=agent_id)
        return {
            'exchange': ExchangeA(config['exchange']),
            'engulf': Engulf(engulf_config)}

    def generate_topology(self, config):
        outer_path = config['outer_path']
        inner_path = config['inner_path']
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'engulf': {
                'trigger': ('trigger',),
                'inner': inner_path,
                'outer': outer_path}}


def test_engulf():
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
        GENERATORS_KEY: [
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
            agent_1_id: {
                GENERATORS_KEY: {
                    'type': ToyAgent,
                    'config': {'agent_id': agent_1_id}
                }
            },
            agent_2_id: {
                GENERATORS_KEY: {
                    'type': ToyAgent,
                    'config': {'agent_id': agent_2_id}
                }
            }
        }
    }

    # configure experiment
    settings = {}
    experiment = compose_experiment(
        hierarchy=hierarchy,
        initial_state=initial_state,
        settings=settings)

    # run simulation
    experiment.update(time_total)
    output = experiment.emitter.get_data()
    experiment.end()  # end required for parallel processes

    # assert that initial agents store has agents 1 & 2,
    # final has only agent 1, and agent 1 subcompartment has 2
    assert [*output[0.0]['agents'].keys()] == ['1', '2']
    assert [*output[10.0]['agents'].keys()] == ['1']
    assert [*output[10.0]['agents']['1']['subcompartments'].keys()] == ['2']

    return output

def run_engulf():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output = test_engulf()
    pp(output)


if __name__ == '__main__':
    run_engulf()
