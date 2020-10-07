"""
===================
Experiment Control
===================

Run experiments and analyses from the command line
"""

import os
import sys
import argparse
import copy

from vivarium.core.experiment import timestamp
from vivarium.core.composition import (
    compartment_hierarchy_experiment,
    simulate_compartment_in_experiment,
    simulate_experiment,
    ToyCompartment,
    BASE_OUT_DIR,
)

from vivarium.plots.simulation_output import plot_simulation_output



def make_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


class Control():

    def __init__(
            self,
            out_dir=None,
            experiments=None,
            compartments=None,
            plots=None,
            workflows=None,
    ):
        if out_dir is None:
            out_dir = timestamp()

        self.experiments = experiments
        self.compartments = compartments
        self.plots = plots
        self.workflows = workflows

        # output directory
        self.base_out_dir = os.path.join(BASE_OUT_DIR, out_dir)
        self.out_dir = self.base_out_dir
        make_dir(self.out_dir)

        # arguments
        self.args = self.add_arguments()

        if self.args.workflow:
            workflow_id = str(self.args.workflow)
            workflow_name = self.workflows[workflow_id].get('name', workflow_id)
            self.out_dir = os.path.join(self.base_out_dir, workflow_name)
            make_dir(self.out_dir)

            self.execute_workflow(workflow_id)

    def add_arguments(self):
        parser = argparse.ArgumentParser(
            description='command line control of experiments'
        )
        parser.add_argument(
            '--workflow', '-w',
            type=str,
            choices=list(self.workflows.keys()),
            help='the workflow id'
        )

        return parser.parse_args()

    def execute_workflow(self, workflow_id):
        workflow = self.workflows[workflow_id]
        experiment_id = workflow['experiment']
        plot_ids = workflow['plots']

        # run the experiment
        experiment = self.experiments[experiment_id]
        if isinstance(experiment, dict):
            data = experiment['experiment']()
        elif callable(experiment):
            data = experiment()

        # run the plots with the data
        if isinstance(plot_ids, list):
            for plot_id in plot_ids:
                plots = self.plots[plot_id]
                data_copy = copy.deepcopy(data)
                plots(
                    data=data_copy,
                    out_dir=self.out_dir)
        else:
            plots = self.plots[plot_ids]
            plots(
                data=data,
                out_dir=self.out_dir)



# testing
def toy_experiment():
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


def toy_plot(data, out_dir='out'):
    plot_simulation_output(data, out_dir=out_dir)


def toy_control():
    experiment_library = {
        '1': {
            'name': 'exp_1',
            'experiment': toy_experiment
        }
    }
    plot_library = {
        '1': toy_plot
    }
    compartment_library = {
        'agent': ToyCompartment,
    }
    workflow_library = {
        '1': {
            'name': 'test_workflow',
            'experiment': '1',
            'plots': ['1'],
        }
    }

    control = Control(
        out_dir='control_test',
        experiments=experiment_library,
        compartments=compartment_library,
        plots=plot_library,
        workflows=workflow_library,
        )

    return control


def test_control():
    control = toy_control()
    control.execute_workflow('1')


if __name__ == '__main__':
    toy_control()
