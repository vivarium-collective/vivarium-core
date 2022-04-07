"""
===============
Burst Process
===============
"""

import os

from vivarium.core.engine import pp
from vivarium.core.process import (
    Step,
)
from vivarium.core.composer import Composer, Composite
from vivarium.core.engine import Engine
from vivarium.core.composition import PROCESS_OUT_DIR
from vivarium.composites.toys import ExchangeA
from vivarium.processes.timeline import TimelineProcess


NAME = 'burst'


class Burst(Step):
    """ Burst Process

    Remove a compartment when the state under the 'trigger' port is set to True.
    Move its inner stores to the outer.
    """
    name = NAME
    defaults = {
        'agent_id': 'DEFAULT'
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.agent_id = self.parameters['agent_id']

    def ports_schema(self):
        return {
            'trigger': {
                '_default': False,
            },
            'inner': {
                '*': {}
            },
            'outer': {
                '*': {}
            },
            'compartment': {
                '*': {}
            },
        }

    def next_update(self, timestep, states):
        if states['trigger']:
            inner_states = states['inner']

            return {
                'inner': {
                    '_move': [{
                        # points to key in 'inner' port
                        'source': state,
                        # points to which port it will be moved
                        'target': 'outer',
                    } for state in inner_states],
                },
                'compartment': {
                    # remove self
                    '_delete': [self.agent_id]
                },
                'trigger': {
                    '_updater': 'set',
                    '_value': False},
            }

        else:
            return {}


# test
class ToyAgent(Composer):
    defaults = {
        'exchange': {
            'uptake_rate': 0.1},
        'inner_path': ('concentrations',),
        'outer_path': ('..', '..', 'concentrations'),
        'compartment_path': ('..', '..', 'agents'),
    }

    def generate_processes(self, config):
        agent_id = config['agent_id']
        return {
            'exchange': ExchangeA(config['exchange']),
            'burst': Burst({'agent_id': agent_id})}

    def generate_topology(self, config):
        return {
            'exchange': {
                'internal': config['inner_path'],
                'external': config['outer_path']},
            'burst': {
                'trigger': ('trigger',),
                'inner': config['inner_path'],
                'outer': config['outer_path'],
                'compartment': config['compartment_path'],
            }}


def test_burst():
    agent_1_id = '1'
    agent_2_id = '2'

    # initial state
    initial_a = 10
    initial_state = {
        'concentrations': {
            'A': initial_a
        },
        'agents': {
            agent_1_id: {
                'trigger': False,
                'concentrations': {
                    'A': 0
                },
                'agents': {
                    agent_2_id: {
                        'trigger': False,
                        'concentrations': {
                            'A': 0
                        },
                    }
                },
            }
        }
    }

    # timeline triggers burst for agent_s
    time_burst = 3
    time_total = 5
    timeline = [
        (0, {('agents', agent_1_id, 'agents', agent_2_id, 'trigger'): False}),
        (time_burst, {('agents', agent_1_id, 'agents', agent_2_id, 'trigger'): True}),
        (time_total, {})]

    # declare the hierarchy
    timeline = TimelineProcess({'timeline': timeline})
    agent1 = ToyAgent({'agent_id': agent_1_id})
    agent2 = ToyAgent({'agent_id': agent_2_id})

    composite = Composite()
    composite.merge(agent1.generate(path=('agents', agent_1_id)))
    composite.merge(agent2.generate(path=('agents', agent_1_id, 'agents', agent_2_id)))
    composite.merge(
        processes={'timeline': timeline},
        topology={'timeline': {
            'global': ('global',),
            'agents': ('agents',)}},
        state=initial_state
    )

    # configure experiment
    experiment = Engine(
        composite=composite,
    )

    pp(experiment.topology)
    pp(experiment.state.get_value())

    # run simulation
    experiment.update(time_total)
    output = experiment.emitter.get_data()
    experiment.end()  # end required for parallel processes

    # asserts total A is the same at the beginning and the end
    assert output[0.0]['concentrations']['A'] == initial_a
    assert output[5.0]['concentrations']['A'] + output[5.0]['agents'][agent_1_id]['concentrations']['A'] == initial_a

    return output


def run_burst():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    os.makedirs(out_dir, exist_ok=True)
    output = test_burst()
    pp(output)


if __name__ == '__main__':
    run_burst()
