"""
===================
Experiment Control
===================

Run experiments and analyses from the command line
"""

import os
import sys
import argparse

from arpeggio import (
    RegExMatch,
    ParserPython,
    OneOrMore,
)

from vivarium.core.experiment import timestamp
from vivarium.core.composition import (
    compartment_hierarchy_experiment,
    simulate_experiment,
    ToyCompartment,
    ToyEnvironment,
    EXPERIMENT_OUT_DIR,
)

from vivarium.plots.agents_multigen import plot_agents_multigen



# parsing expression grammar for agents
def agent_type(): return RegExMatch(r'[a-zA-Z0-9.\-\_]+')
def number(): return RegExMatch(r'[0-9]+')
def specification(): return agent_type, number
def rule(): return OneOrMore(specification)
agent_parser = ParserPython(rule)

def parse_agents_string(agents_string):
    all_agents = agent_parser.parse(agents_string)
    agents_config = []
    for idx, agent_specs in enumerate(all_agents):
        agents_config.append(make_agent_config(agent_specs))
    return agents_config


def make_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


class Control():

    def __init__(
            self,
            name=None,
            experiment_library={},
            plot_library={},
            # compartment_library={},
            # workflows={},
            # simulation_settings={},
    ):
        if name is None:
            name = timestamp()
        # self.compartment_library = compartment_library
        self.experiment_library = experiment_library
        self.plot_library = plot_library
        self.args = self.add_arguments()

        # TODO experiment settings
        # TODO plot settings

        self.out_dir = os.path.join(EXPERIMENT_OUT_DIR, name)
        make_dir(self.out_dir)

        import ipdb; ipdb.set_trace()

    def add_arguments(self):
        parser = argparse.ArgumentParser(
            description='command line control of experiments'
        )
        parser.add_argument(
            '--agents', '-a',
            type=str,
            nargs='+',
            default=argparse.SUPPRESS,
            help='A list of agent types and numbers in the format "agent_type1 number1 agent_type2 number2"'
        )
        parser.add_argument(
            '--environment', '-v',
            type=str,
            default=argparse.SUPPRESS,
            help='the environment type'
        )
        parser.add_argument(
            '--time', '-t',
            type=int,
            default=60,
            help='simulation time, in seconds'
        )
        parser.add_argument(
            '--emit', '-m',
            type=int,
            default=1,
            help='emit interval, in seconds'
        )
        parser.add_argument(
            '--experiment', '-e',
            type=str,
            default=argparse.SUPPRESS,
            help='preconfigured experiments'
        )

        return vars(parser.parse_args())

    def execute(self):
        if self.args['experiment']:
            experiment_name = self.args['experiment']
            experiment_out_dir = os.path.join(self.out_dir, experiment_name)
            make_dir(experiment_out_dir)
            experiment_config = self.experiment_library[experiment_name]
            hierarchy = experiment_config['hierarchy']
            simulation_settings = experiment_config['simulation_settings']

        # simulate
        data = self.run_experiment(
            hierarchy=hierarchy,
            # initial_state=initial_state,
            # initial_agent_state=initial_agent_state,
            simulation_settings=simulation_settings,
        )


    def run_experiment(
            self,
            # agents_config=None,
            # environment_config=None,
            initial_state=None,
            # initial_agent_state=None,
            hierarchy=None,
            simulation_settings=None,
            experiment_settings=None
    ):
        if experiment_settings is None:
            experiment_settings = {}
        if initial_state is None:
            initial_state = {}
        # if initial_agent_state is None:
        #     initial_agent_state = {}


        # make the experiment
        experiment = embedded_compartment_experiment(hierarchy)

        # simulate
        settings = {
            'total_time': simulation_settings['total_time'],
            'emit_step': simulation_settings['emit_step'],
            'return_raw_data': simulation_settings['return_raw_data']}
        return simulate_experiment(
            experiment,
            settings,
        )


def experiment_1():
    toy_compartment = ToyCompartment({})
    settings = {
        'total_time': 10,
        'initial_state': {
            'periplasm': {
                'GLC': 20,
                'MASS': 100,
                'DENSITY': 10},
            'cytoplasm': {
                'GLC': 0,
                'MASS': 3,
                'DENSITY': 10}}}
    return simulate_compartment_in_experiment(toy_compartment, settings)

def plot_library():
    pass


def run_control_test():
    experiment_library = {
        '1': experiment_1
    }
    plot_library = {
        '1': plots_1
    }

    workflow = Control(
        experiment_library=experiment_library,
        plot_library=plot_library,
        )

    workflow.execute()


if __name__ == '__main__':
    run_control_test()