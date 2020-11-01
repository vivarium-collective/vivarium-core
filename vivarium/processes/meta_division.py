from __future__ import absolute_import, division, print_function

import os
import uuid
import logging as log

from vivarium.core.process import (
    Deriver,
    Generator
)
from vivarium.core.composition import (
    compose_experiment,
    GENERATORS_KEY,
    PROCESS_OUT_DIR,
)

# processes
from vivarium.processes.exchange_a import ExchangeA
from vivarium.processes.timeline import TimelineProcess

NAME = 'meta_division'


# functions for generating daughter ids
def daughter_uuid(mother_id):
    return [
        str(uuid.uuid1()),
        str(uuid.uuid1())]


def daughter_phylogeny_id(mother_id):
    return [
        str(mother_id) + '0',
        str(mother_id) + '1']


def divider_set_false(state):
    return [False, False]


class MetaDivision(Deriver):
    name = NAME
    defaults = {
        'initial_state': {},
        'daughter_path': ('agents',),
        'daughter_ids_function': daughter_phylogeny_id}

    def __init__(self, initial_parameters=None):
        if initial_parameters is None:
            initial_parameters = {}

        self.division = 0

        # must provide a compartment to generate new daughters
        self.agent_id = initial_parameters['agent_id']
        self.compartment = initial_parameters['compartment']
        self.daughter_ids_function = self.or_default(
            initial_parameters, 'daughter_ids_function')
        self.daughter_path = self.or_default(
            initial_parameters, 'daughter_path')

        super(MetaDivision, self).__init__(initial_parameters)

    def ports_schema(self):
        return {
            'global': {
                'divide': {
                    '_default': False,
                    '_updater': 'set',
                    '_divider': divider_set_false,
                }},
            'agents': {
                '*': {}}}

    def next_update(self, timestep, states):
        divide = states['global']['divide']

        if divide:
            daughter_ids = self.daughter_ids_function(self.agent_id)
            daughter_updates = []

            for daughter_id in daughter_ids:
                compartment = self.compartment.generate({
                    'agent_id': daughter_id})
                daughter_updates.append({
                    'daughter': daughter_id,
                    'path': (daughter_id,) + self.daughter_path,
                    'processes': compartment['processes'],
                    'topology': compartment['topology'],
                    'initial_state': {}})

            log.info(
                'DIVIDE! \n--> MOTHER: {} \n--> DAUGHTERS: {}'.format(
                    self.agent_id, daughter_ids))

            # initial state will be provided by division in the tree
            return {
                'agents': {
                    '_divide': {
                        'mother': self.agent_id,
                        'daughters': daughter_updates}}}
        else:
            return {}


# test
class ToyAgent(Generator):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'daughter_path': tuple(),
        'agents_path': ('..', '..', 'agents')}

    def generate_processes(self, config):
        daughter_path = config['daughter_path']
        agent_id = config['agent_id']
        division_config = dict(
            {},
            daughter_path=daughter_path,
            agent_id=agent_id,
            compartment=self)

        return {
            'exchange': ExchangeA(config['exchange']),
            'division': MetaDivision(division_config)}

    def generate_topology(self, config):
        agents_path = config['agents_path']
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'division': {
                'global': ('global',),
                'agents': agents_path,
            }
        }


def test_division():
    agent_id = '1'

    # initial state
    initial_state = {
        'agents': {
            agent_id: {
                'internal': {'A': 0},
                'external': {'A': 1},
                'global': {
                    'divide': False
                }
            },
        }
    }

    # timeline triggers division
    time_divide = 5
    time_total = 10
    timeline = [
        (0, {('agents', agent_id, 'global', 'divide'): False}),
        (time_divide, {('agents', agent_id, 'global', 'divide'): True}),
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
            agent_id: {
                GENERATORS_KEY: {
                    'type': ToyAgent,
                    'config': {'agent_id': agent_id}
                },
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
    experiment.end()

    # external starts at 1, goes down until death, and then back up
    # internal does the inverse
    assert list(output[time_divide]['agents'].keys()) == [agent_id]
    assert agent_id not in list(output[time_divide + 1]['agents'].keys())
    assert len(output[time_divide]['agents']) == 1
    assert len(output[time_divide + 1]['agents']) == 2

    return output


def run_division():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output = test_division()
    # pp(output)


if __name__ == '__main__':
    run_division()
