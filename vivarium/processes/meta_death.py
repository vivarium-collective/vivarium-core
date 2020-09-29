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
from vivarium.plots.simulation_output import plot_simulation_output


NAME = 'meta_death'


class MetaDeath(Deriver):
    """ MetaDeath Process

    Declared death processes replace the contents of the compartment
    when the state under the 'dead' port is set to True.
    """
    name = NAME
    defaults = {
        'initial_state': {},   # TODO -- get this from the compartment
        'death_paths': [],  # list of processes to remove
    }

    def __init__(self, parameters=None):
        super(MetaDeath, self).__init__(parameters)
        self.death_paths = self.parameters['death_paths']
        self.death_compartment = self.parameters['death_compartment']
        self.initial_state = self.parameters['initial_state']

    def ports_schema(self):
        return {
            'dead': {
                '_default': False,
                '_emit': True},
            'self': {}}

    def next_update(self, timestep, states):
        if states['dead']:
            network = self.death_compartment.generate({})  # todo -- pass in config?
            return {
                'self': {
                    '_delete': self.death_paths,
                    '_generate': [{
                        'processes': network['processes'],
                        'topology': network['topology'],
                        'initial_state': self.initial_state,
                    }]
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
        'secrete': {
            'secrete_rate': 0.1}}

    def generate_processes(self, config):
        return {
            'secrete': ExchangeA(config['secrete'])}

    def generate_topology(self, config):
        return {
            'secrete': {
                'internal': ('internal',),
                'external': ('external',)}}


class ToyLivingCompartment(Generator):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'death': {
            'death_paths': [
                ('exchange',),
                ('death',)],
            'death_compartment': ToyDeadCompartment({})
        }}

    def generate_processes(self, config):
        return {
            'exchange': ExchangeA(config['exchange']),
            'death': MetaDeath(config['death'])}

    def generate_topology(self, config):
        self_path = ('..', config['agent_id'])
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'death': {
                'dead': ('dead',),
                'self': self_path}}


def test_death():
    agent_id = '1'

    # make the compartment
    compartment = ToyLivingCompartment({
        'agent_id': agent_id})

    # initial state
    initial_state = {
        'agents': {
            agent_id: {
                'external': {'A': 1},
                'dead': False}}}

    # timeline turns death on
    timeline = [
        (0, {('agents', agent_id, 'dead'): False}),
        (5, {('agents', agent_id, 'dead'): True}),
        (10, {})]

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
    plot_simulation_output(output, {}, out_dir)


if __name__ == '__main__':
    run_death()
