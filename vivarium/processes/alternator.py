"""
===============
Alternator Process
===============
"""

import os

from vivarium.core.engine import pp
from vivarium.core.process import (
    Process,
)
from vivarium.core.composer import Composer
from vivarium.core.composition import (
    compose_experiment,
    COMPOSER_KEY,
    PROCESS_OUT_DIR,
)
from vivarium.composites.toys import ExchangeA
from vivarium.processes.timeline import TimelineProcess


NAME = 'alternator'


def find_chosen(choices):
    '''
    given a dict of choices key -> True/False where only one value is True,
    return the index of the True value. 
    '''

    for index, option in enumerate(choices.items()):
        choice, chosen = option
        if chosen:
            return choice, index
    return None, -1


def choose_option(choices, chosen_index):
    '''
    given a list of choices and an index, construct a dict where the keys are the choices
    and the values are False everywhere except the given index
    '''

    return {
        choice: index == chosen_index
        for index, choice in enumerate(choices)}


class Alternator(Process):
    """ Alternator Process

    Alternate between different choices for the given periods of time. 
    """

    name = NAME
    defaults = {
        'choices': [],
        'periods': [],
    }

    def __init__(self, parameters=None):
        super(Alternator, self).__init__(parameters)
        self.choices = self.parameters['choices']
        self.periods = self.parameters['periods']

    def choice_index(self, choice):
        for index, option in enumerate(self.parameters['choices']):
            if choice == option:
                return index
        return -1

    def calculate_timestep(self, states):
        period = 0
        num_periods = len(self.parameters['periods'])
        for index, choice in enumerate(self.parameters['choices']):
            if states[choice]:
                period = self.parameters['periods'][index % num_periods]
                break
        return period

    def initial_state(self, config):
        return {
            choice: index == 0
            for index, choice in enumerate(self.parameters['choices'])}

    def ports_schema(self):
        choices = {
            choice: {
                '_default': False}
            for choice in self.parameters['choices']}

        return choices

    def next_update(self, timestep, states):
        chosen, index = find_chosen(states)
        next_choice = (index + 1) % len(self.parameters['choices'])

        import ipdb; ipdb.set_trace()

        return choose_option(self.parameters['choices'], next_choice)


# test
class ExchangeAlternator(Composer):
    defaults = {
        'on_exchange': {
            'uptake_rate': 0.2,
            '_condition': 'ON'},
        'off_exchange': {
            'uptake_rate': 0.5,
            '_condition': 'OFF'},
        'alternator': {
            'choices': ['ON', 'OFF'],
            'periods': [10.0, 4.0]}}

    def generate_processes(self, config):
        return {
            'on_exchange': ExchangeA(config['on_exchange']),
            'off_exchange': ExchangeA(config['off_exchange']),
            'alternator': Alternator(config['alternator'])}

    def generate_topology(self, config):
        return {
            'on_exchange': {
                'internal': ('concentrations', 'ON'),
                'external': ('concentrations', 'OFF')},
            'off_exchange': {
                'internal': ('concentrations', 'OFF'),
                'external': ('concentrations', 'ON')},
            'alternator': {
                'ON': ('ON',),
                'OFF': ('OFF',)}}


def test_alternator():
    initial_on = 10.0
    initial_off = 13.0

    initial_state = {
        'concentrations': {
            'ON': {
                'A': initial_on},
            'OFF': {
                'A': initial_off}
        },
        'ON': False,
        'OFF': True,
    }

    # declare the hierarchy
    hierarchy = {
        COMPOSER_KEY: {
                'type': ExchangeAlternator,
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

    # timeline triggers alternator for agent_s
    time_total = 25.0

    # run simulation
    experiment.update(time_total)
    output = experiment.emitter.get_data()
    experiment.end()  # end required for parallel processes

    import ipdb; ipdb.set_trace()

    # asserts total A is the same at the beginning and the end
    assert output[0.0]['concentrations']['A'] == initial_a
    assert output[5.0]['concentrations']['A'] + output[5.0]['agents'][agent_1_id]['concentrations']['A'] == initial_a

    return output


def run_alternator():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    os.makedirs(out_dir, exist_ok=True)
    output = test_alternator()
    pp(output)


if __name__ == '__main__':
    run_alternator()
