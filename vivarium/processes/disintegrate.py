"""
====================
Disintegrate Process
====================
"""

import os

from vivarium.core.experiment import pp
from vivarium.core.process import (
    Deriver,
    Generator,
)
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.processes.exchange_a import ExchangeA


NAME = 'disintegrate'


class Disintegrate(Deriver):
    """ Disintegrate Process

    remove a compartment when the state under the 'trigger' port is set to True.
    """
    name = NAME
    defaults = {}

    def __init__(self, parameters=None):
        super(Disintegrate, self).__init__(parameters)
        self.agent_id = self.parameters['agent_id']

    def ports_schema(self):
        return {
            'trigger': {
                '_default': False,
                '_emit': True},
            'agents': {}}

    def next_update(self, timestep, states):
        if states['trigger']:
            return {
                'agents': {
                    '_delete': [(self.agent_id,)]}}
        else:
            return {}


# test
class ToyLivingCompartment(Generator):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'death': {}
    }

    def generate_processes(self, config):
        death_config = config['death']
        death_config['agent_id'] = config['agent_id']
        return {
            'exchange': ExchangeA(config['exchange']),
            'death': Disintegrate(death_config)}

    def generate_topology(self, config):
        agents_path = ('..', '..', 'agents')
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'death': {
                # set the trigger to be the 'dead' state
                'trigger': ('dead',),
                'agents': agents_path}}


def test_disintegrate():
    agent_id = '1'

    # make the compartment
    compartment = ToyLivingCompartment({
        'agent_id': agent_id})

    # initial state
    initial_state = {
        'agents': {
            agent_id: {
                'external': {'A': 1},
                'trigger': False}}}

    # timeline turns death on
    time_dead = 5
    time_total = 10
    timeline = [
        (0, {('agents', agent_id, 'dead'): False}),
        (time_dead, {('agents', agent_id, 'dead'): True}),
        (time_total, {})]

    # simulate
    settings = {
        'outer_path': ('agents', agent_id),
        'timeline': {
            'timeline': timeline},
        'initial_state': initial_state}
    output = simulate_compartment_in_experiment(
        compartment,
        settings)

    assert len(output['agents']['1']['dead']) == time_dead + 1
    assert len(output['time']) == time_total + 1

    return output

def run_disintegrate():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output = test_disintegrate()
    pp(output)


if __name__ == '__main__':
    run_disintegrate()
