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
    when the global variable 'die' is triggered.
    """
    name = NAME
    defaults = {
        'initial_state': {},
        'death_processes': {},
        'death_topology': {},
    }

    def __init__(self, parameters=None):
        super(MetaDeath, self).__init__(parameters)
        self.compartment = self.parameters['compartment']
        self.death_processes = self.parameters['death_processes']

        # make topology
        self.death_topology = {}
        for process_id, process in self.death_processes.items():
            ports = process.ports()
            self.death_topology[process_id] = {
                port: (port,) for port in ports.keys()}
        self.death_topology.update(self.parameters['death_topology'])

        self.initial_state = self.parameters['initial_state']

    def ports_schema(self):
        return {
            'global': {
                'die': {
                    '_default': False,
                    '_updater': 'set',
                    '_emit': True,
                },
                'has_died': {
                    '_default': False,
                    '_updater': 'set',
                }
            },
            'compartment': {}
        }

    def next_update(self, timestep, states):
        die = states['global']['die']
        has_died = states['global']['has_died']
        if die and not has_died:
            # Get processes to remove from compartment
            network = self.compartment.generate({})
            living_processes = network['processes'].keys()

            update = {
                'global': {
                    'has_died': True},
                'compartment': {
                    '_delete': [
                        (processes,)
                        for processes in living_processes],
                    '_generate': [{
                        'processes': self.death_processes,
                        'topology': self.death_topology,
                        'initial_state': self.initial_state}]
                }
            }
            return update
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

class ToyLivingCompartment(Generator):
    exchange_topology = {
        'internal': ('internal',),
        'external': ('external',)}
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'death': {
            'death_processes': {
                'secretion': ExchangeA({
                    'exchange': {'uptake_rate': -0.1}})},
            'death_topology': {
                'secretion': exchange_topology}}}

    def generate_processes(self, config):
        agent_id = config['agent_id']
        death_config = config['death']
        death_config.update({
            'agent_id': agent_id,
            'compartment': self})
        return {
            'exchange': ExchangeA(config['exchange']),
            'death': MetaDeath(death_config)}

    def generate_topology(self, config):
        self_path = ('..', config['agent_id'])
        return {
            'exchange': self.exchange_topology,
            'death': {
                'global': ('global',),
                'compartment': self_path}}


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
                'global': {'dead': False}
            }
        }
    }

    # timeline turns death on
    timeline = [
        (0, {('agents', agent_id, 'global', 'die'): False}),
        (5, {('agents', agent_id, 'global', 'die'): True}),
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

    import ipdb;
    ipdb.set_trace()


if __name__ == '__main__':
    run_death()
