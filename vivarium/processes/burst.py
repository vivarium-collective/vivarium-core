"""
===============
Burst Process
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


NAME = 'burst'


class Burst(Deriver):
    """ Burst Process

    Remove a compartment when the state under the 'trigger' port is set to True.
    Move its inner stores to the outer.
    """
    name = NAME
    defaults = {
        'inner_path': ('inner',)
    }

    def __init__(self, parameters=None):
        super(Burst, self).__init__(parameters)
        self.agent_id = self.parameters['agent_id']
        self.inner_path = self.parameters['inner_path']

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
            }
        }

    def next_update(self, timestep, states):
        if states['trigger']:
            inner_states = states['inner']
            return {
                'outer': {
                    # move each inner state to outer
                    '_move': [{
                        'source': (self.agent_id,) + self.inner_path + (state,),
                        'target': tuple(),
                    } for state in inner_states],
                    # remove self
                    '_delete': [(self.agent_id,)],
                },
                'trigger': {
                    '_updater': 'set',
                    '_value': False},
            }

        else:
            return {}


# test
class ToyAgent(Generator):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
    }

    def generate_processes(self, config):
        agent_id = config['agent_id']
        return {
            'exchange': ExchangeA(config['exchange']),
            'burst': Burst({'agent_id': agent_id})}

    def generate_topology(self, config):
        return {
            'exchange': {
                'internal': ('inner',),
                'external': ('outer',)},
            'burst': {
                'trigger': ('trigger',),
                'inner': ('inner',),
                'outer': ('outer',)}}


def test_burst():
    agent_1_id = '1'
    agent_2_id = '2'

    # initial state
    initial_A = 10
    initial_state = {
        'agents': {
            agent_1_id: {
                'outer': {'A': initial_A},
                'inner': {
                    'A': 0,
                    agent_2_id: {
                        'inner': {'A': 0},
                        'trigger': []
                    }
                },
                'trigger': False,
            }
        }
    }

    # timeline triggers burst for agent_s
    time_burst = 3
    time_total = 5
    timeline = [
        (0, {('agents', agent_1_id, 'inner', agent_2_id, 'trigger'): False}),
        (time_burst, {('agents', agent_1_id, 'inner', agent_2_id, 'trigger'): True}),
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
                },
                'inner': {
                    agent_2_id: {
                        GENERATORS_KEY: {
                            'type': ToyAgent,
                            'config': {'agent_id': agent_2_id},
                            'topology': {
                                'exchange': {
                                    'external': ('..', '..', 'inner')},
                                'burst': {
                                    'outer': ('..', '..', 'inner')},
                            }
                        }
                    }
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

    pp(experiment.topology)
    pp(experiment.state.get_value())

    # run simulation
    experiment.update(time_total)
    output = experiment.emitter.get_data()
    experiment.end()  # end required for parallel processes

    # asserts total A is the same at the beginning and the end
    assert output[0.0]['agents'][agent_1_id]['outer']['A'] == initial_A
    assert output[5.0]['agents'][agent_1_id]['outer']['A'] + output[5.0]['agents'][agent_1_id]['inner']['A'] == initial_A

    return output

def run_burst():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output = test_burst()
    pp(output)



if __name__ == '__main__':
    run_burst()
