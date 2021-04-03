"""
===============
Engulf Process
===============
"""

import os

from vivarium.core.experiment import pp
from vivarium.core.process import (
    Deriver,
    Composer,
)
from vivarium.core.composition import (
    compose_experiment,
    COMPOSER_KEY,
    PROCESS_OUT_DIR,
)
from vivarium.composites.toys import ExchangeA
from vivarium.processes.timeline import TimelineProcess


NAME = 'engulf'


class Engulf(Deriver):
    """ Engulf Process

    remove a compartment when the state under the 'trigger' port is set to True.
    """
    name = NAME
    defaults = {
        'agent_id': 'DEFAULT'}

    def __init__(self, parameters=None):
        super(Engulf, self).__init__(parameters)
        self.agent_id = self.parameters['agent_id']

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
            neighbors = states['trigger']
            # move neighbors from outer to inner, reset trigger
            return {
                'trigger': {
                    '_updater': 'set',
                    '_value': []},
                'outer': {
                    '_move': [{
                        # points to key in 'outer' port
                        'source': neighbor,
                        # points to which port it will be moved
                        'target': 'inner'
                    } for neighbor in neighbors]
                }
            }
        else:
            return {}


# test
class ToyAgent(Composer):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'engulf': {
            'outer_path': ('..', '..', 'agents'),
            'inner_path': ('agents',)}}

    def generate_processes(self, config):
        return {
            'exchange': ExchangeA(config['exchange']),
            'engulf': Engulf(config['engulf']),
            'expel': Engulf(config['engulf'])}

    def generate_topology(self, config):
        return {
            'exchange': {
                'external': config['exchange']['external_path'],
                'internal': config['exchange']['internal_path']},
            'engulf': {
                'trigger': ('engulf-trigger',),
                'inner': config['engulf']['inner_path'],
                'outer': config['engulf']['outer_path']},
            'expel': {
                'trigger': {
                    '_path': ('expel-trigger',)},
                'inner': config['engulf']['outer_path'],
                'outer': config['engulf']['inner_path']}}


def test_engulf():
    num_agents = 3
    agent_ids = [
        str(agent_id + 1)
        for agent_id in range(num_agents)]

    # initial state
    initial_state = {
        'concentrations': {'A': 10.0},
        'agents': {
            agent_id: {
                'concentrations': {'A': float(int(agent_id))},
                'trigger': []}
            for agent_id in agent_ids}}

    # timeline triggers engulf for agent_1
    time_total = 10
    timeline = [
        (0, {('agents', agent_ids[0], 'trigger'): []}),
        (3, {('agents', agent_ids[2], 'engulf-trigger'): [agent_ids[1]]}),
        (5, {('agents', agent_ids[0], 'engulf-trigger'): [agent_ids[2]]}),
        (8, {('agents', agent_ids[0], 'agents', agent_ids[2], 'expel-trigger'): [agent_ids[1]]}),
        (time_total, {})]

    # declare the hierarchy
    hierarchy = {
        COMPOSER_KEY: [
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
            agent_id: {
                COMPOSER_KEY: {
                    'type': ToyAgent,
                    'config': {
                        'exchange': {
                            'internal_path': ('concentrations',),
                            'external_path': ('..', '..', 'concentrations')},
                        'engulf': {
                            'inner_path': ('agents',),
                            'outer_path': ('..', '..', 'agents')}}
                }
            } for agent_id in agent_ids
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

    # assert the engulfing/expelling saga
    assert [*output[0.0]['agents'].keys()] == agent_ids
    assert [*output[4.0]['agents'].keys()] == ['1', '3']
    assert [*output[6.0]['agents']['1']['agents'].keys()] == ['3']
    assert [*output[10.0]['agents'].keys()] == ['1']
    assert [*output[10.0]['agents']['1']['agents'].keys()] == ['3', '2']

    return output


def run_engulf():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    os.makedirs(out_dir, exist_ok=True)
    output = test_engulf()
    pp(output)


if __name__ == '__main__':
    run_engulf()
