"""
=====================
General Death Process
=====================
"""

import os
import uuid
import logging as log

from vivarium.core.process import (
    Deriver,
    Process,
    Generator,
)
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.plots.plot import plot_agents_multigen


NAME = 'meta_death'


class MetaDeath(Deriver):
    name = NAME
    defaults = {
        'initial_state': {},
        'dead_compartment': None,
    }

    def __init__(self, parameters=None):
        super(MetaDeath, self).__init__(parameters)

        # provide dead_compartment to replace
        # current compartment after death
        self.agent_id = self.parameters['agent_id']
        self.dead_compartment = self.parameters['dead_compartment']

    def ports_schema(self):
        return {
            'global': {
                'dead': {
                    '_default': False,
                    '_updater': 'set',
                    '_emit': True,
                }
            },
            'agents': {
                '*': {}
            }
        }

    def next_update(self, timestep, states):
        dead = states['global']['dead']

        if dead:
            compartment = self.dead_compartment.generate({
                'agent_id': self.agent_id + '_dead'})

            log.info('DEATH! {}'.format(self.agent_id))

            return {
                'agents': {
                    '_delete': self.agent_id,
                    '_add': compartment,
                }
            }
        else:
            return {}


# test
class ExchangeA(Process):
    name = 'exchange_a'
    defaults = {
        'uptake_rate': 0.0,
        'secrete_rate': 0.0}

    def __init__(self, parameters=None):
        super(ExchangeA, self).__init__(parameters)
        self.uptake_rate = self.parameters['uptake_rate']
        self.secrete_rate = self.parameters['secrete_rate']

    def ports_schema(self):
        return {
            'internal': {
                'A': {
                    '_default': 0.0,
                    '_emit': True}},
            'external': {
                'A': {
                    '_default': 0.0,
                    '_emit': True}}}

    def next_update(self, timestep, states):
        A_in = states['internal']['A']
        A_out = states['external']['A']
        delta_A_in = A_out * self.uptake_rate - A_in * self.secrete_rate
        return {
            'internal': {'A': delta_A_in},
            'external': {'A': -delta_A_in}}

class ToyDeadCompartment(Generator):
    defaults = {
        'agents_path': ('agents',),
        'exchange': {
            'uptake_rate': -0.1}}

    def generate_processes(self, config):
        return {'exchange': ExchangeA(config['exchange']),}

    def generate_topology(self, config):
        agents_path = config['agents_path']
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)}}

class ToyLivingCompartment(Generator):
    defaults = {
        'agents_path': ('agents',),
        'exchange': {'uptake_rate': 0.1},
        'death': {'dead_compartment': ToyDeadCompartment({})}}

    def generate_processes(self, config):
        return {
            'exchange': ExchangeA(config['exchange']),
            'death': MetaDeath(config['death'])}

    def generate_topology(self, config):
        agents_path = config['agents_path']
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'death': {
                'global': ('global',),
                'agents': agents_path}}


def test_death():
    agent_id = '1'

    # make the compartment
    compartment = ToyLivingCompartment({
        'agents_path': ('..', '..', 'agents'),
        'death': {'agent_id': agent_id}})

    # initial state
    initial_state = {
        'agents': {
            agent_id: {
                'external': {'A': 1},
                'global': {'dead': 0}
            }
        }
    }

    # timeline turns death on
    timeline = [
        (0, {('agents', agent_id, 'global', 'dead'): False}),
        (2, {('agents', agent_id, 'global', 'dead'): True}),
        (4, {})]

    # simulate
    settings = {
        'outer_path': ('agents', agent_id),
        'timeline': {
            'timeline': timeline},
        'initial_state': initial_state}
    return simulate_compartment_in_experiment(
        compartment,
        settings)


def run_death():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output = test_death()

    import ipdb;
    ipdb.set_trace()

    plot_agents_multigen(output, {}, out_dir)


if __name__ == '__main__':
    run_death()
