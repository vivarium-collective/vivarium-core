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
    simulate_compartment_in_experiment,
    simulate_experiment,
    ToyCompartment,
    ToyEnvironment,
    EXPERIMENT_OUT_DIR,
)

from vivarium.plots.agents_multigen import plot_agents_multigen
from vivarium.plots.simulation_output import plot_simulation_output


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
            experiment_library=None,
            plot_library=None,
            workflows=None,
    ):
        if name is None:
            name = timestamp()

        self.experiment_library = experiment_library
        self.plot_library = plot_library
        self.workflows = workflows
        self.args = self.add_arguments()

        self.out_dir = os.path.join(EXPERIMENT_OUT_DIR, name)
        make_dir(self.out_dir)

    def add_arguments(self):
        parser = argparse.ArgumentParser(
            description='command line control of experiments'
        )
        parser.add_argument(
            '--workflow', '-w',
            type=str,
            default=argparse.SUPPRESS,
            help='the workflow name'
        )
        # parser.add_argument(
        #     '--agents', '-a',
        #     type=str,
        #     nargs='+',
        #     default=argparse.SUPPRESS,
        #     help='A list of agent types and numbers in the format "agent_type1 number1 agent_type2 number2"'
        # )
        # parser.add_argument(
        #     '--environment', '-v',
        #     type=str,
        #     default=argparse.SUPPRESS,
        #     help='the environment type'
        # )
        # parser.add_argument(
        #     '--time', '-t',
        #     type=int,
        #     default=60,
        #     help='simulation time, in seconds'
        # )
        # parser.add_argument(
        #     '--emit', '-m',
        #     type=int,
        #     default=1,
        #     help='emit interval, in seconds'
        # )
        # parser.add_argument(
        #     '--experiment', '-e',
        #     type=str,
        #     default=argparse.SUPPRESS,
        #     help='preconfigured experiments'
        # )

        return vars(parser.parse_args())

    def execute_workflow(self, name):
        workflow = self.workflows[name]
        experiment = self.experiment_library[workflow['experiment']]
        plots = self.plot_library[workflow['plots']]

        # run the experiment
        data = experiment()

        # plot
        plots(
            data=data,
            out_dir=self.out_dir)




# testing
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


def plots_1(data, out_dir='out'):
    plot_simulation_output(data, out_dir=out_dir)


def test_control():
    experiment_library = {
        '1': experiment_1
    }
    plot_library = {
        '1': plots_1
    }
    workflows = {
        '1': {
            'experiment': '1',
            'plots': '1',
        }
    }

    control = Control(
        name='control_test',
        experiment_library=experiment_library,
        plot_library=plot_library,
        workflows=workflows,
        )

    control.execute_workflow('1')


if __name__ == '__main__':
    test_control()