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
        # current compartment with after death
        self.agent_id = self.parameters['agent_id']
        self.dead_compartment = self.parameters['dead_compartment']

    def ports_schema(self):
        return {
            'global': {
                'dead': {
                    '_default': False,
                    '_updater': 'set'
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
        'exchange': {
            'uptake_rate': -0.1}}

    def generate_processes(self, config):
        return {'exchange': ExchangeA(config['exchange']),}

    def generate_topology(self, config):
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)}}

class ToyLivingCompartment(Generator):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'death': {'dead_compartment': ToyDeadCompartment({})}}

    def generate_processes(self, config):
        return {
            'exchange': ExchangeA(config['exchange']),
            'death': MetaDeath(config['death'])}

    def generate_topology(self, config):
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'death': {
                'global': ('global',)}}


def test_death():
    # make the compartment
    compartment = ToyLivingCompartment({
        'death': {'agent_id': '1'}})

    # intitial state
    initial_state = {
        'external': {'A': 1},
        'global': {'dead': 0}}

    # timeline turns death on
    timeline = [
        (0, {('global', 'dead'): 0}),
        (10, {('global', 'dead'): 1}),
        (20, {})]

    # simulate
    settings = {
        # 'total_time': 10,
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
    plot_agents_multigen(output, {}, out_dir)


if __name__ == '__main__':
    run_death()
