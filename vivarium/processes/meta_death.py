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
from vivarium.plots.plot import plot_simulation_output


NAME = 'meta_death'


class MetaDeath(Deriver):
    name = NAME
    defaults = {
        'initial_state': {},
        'death_processes': {},
    }

    def __init__(self, parameters=None):
        super(MetaDeath, self).__init__(parameters)

        # provide dead_compartment to replace
        # current compartment after death
        self.agent_path = ('..', self.parameters['agent_id'])
        self.compartment = self.parameters['compartment']
        self.death_processes = self.parameters['death_processes']

    def ports_schema(self):
        return {
            'global': {
                'die': {
                    '_default': False,
                    '_updater': 'set',
                    '_emit': True,
                },
                'already_dead': {
                    '_default': False,
                    '_updater': 'set',
                }
            },
        }

    def next_update(self, timestep, states):
        die = states['global']['die']
        already_dead = states['global']['already_dead']
        if die and not already_dead:
            # Get processes to remove from compartment
            network = self.compartment.generate({})
            living_processes = network['processes'].keys()

            update = {
                'global': {'already_dead': True}
                self.agent_path: {
                    '_delete': [
                        (processes,)
                        for processes in living_processes
                    ],
                    '_generate': [
                        {
                            'path': tuple(),
                            'processes': self.death_processes,
                            'topology': {
                                process.name: {
                                    port: (port,)
                                    for port in process.ports().keys()}
                                for process in self.death_processes
                            },
                        }
                    ],
                },
            }

            import ipdb;
            ipdb.set_trace()

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
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'death': {
            'death_processes': [ExchangeA({
                'exchange': {'uptake_rate': -0.1}})]
        }
    }

    def generate_processes(self, config):
        death_config = config['death']
        death_config.update({'compartment': self})
        return {
            'exchange': ExchangeA(config['exchange']),
            'death': MetaDeath(death_config)}

    def generate_topology(self, config):
        # agents_path = config['agents_path']
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'death': {
                'global': ('global',),
            }}


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
                'global': {'dead': False}
            }
        }
    }

    # timeline turns death on
    timeline = [
        # (0, {('agents', agent_id, 'global', 'die'): False}),
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

    import ipdb;
    ipdb.set_trace()

    plot_simulation_output(output, {}, out_dir)


if __name__ == '__main__':
    run_death()
