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
        if workflows is None:
            workflows = {}
        if experiments is None:
            experiments = {}

        self.experiments_library = experiments
        self.compartments_library = compartments
        self.plots_library = plots
        self.workflows_library = workflows
        self.output_data = None

        # output directory
        self.base_out_dir = os.path.join(BASE_OUT_DIR, out_dir)
        self.out_dir = self.base_out_dir
        make_dir(self.out_dir)

        # arguments
        self.args = self.add_arguments()

        if self.args.experiment:
            experiment_id = str(self.args.experiment)
            self.output_data = self.run_experiment(experiment_id)

        if self.args.workflow:
            workflow_id = str(self.args.workflow)
            workflow_name = self.workflows_library[workflow_id].get('name', workflow_id)
            self.out_dir = os.path.join(self.base_out_dir, workflow_name)
            make_dir(self.out_dir)

            self.run_workflow(workflow_id)

    def add_arguments(self):
        parser = argparse.ArgumentParser(
            description='command line control of experiments'
        )
        parser.add_argument(
            '--workflow', '-w',
            type=str,
            choices=list(self.workflows_library.keys()),
            help='the workflow id'
        )
        parser.add_argument(
            '--experiment', '-e',
            type=str,
            choices=list(self.experiments_library.keys()),
            help='experiment id to run'
        )
        return parser.parse_args()

    def run_experiment(self, experiment_id):
        experiment = self.experiments_library[experiment_id]
        if isinstance(experiment, dict):
            return experiment['experiment']()
        elif callable(experiment):
            return experiment()

    def run_plots(self, plot_ids, data):
        if isinstance(plot_ids, list):
            for plot_id in plot_ids:
                plots = self.plots_library[plot_id]
                data_copy = copy.deepcopy(data)
                plots(
                    data=data_copy,
                    out_dir=self.out_dir)
        else:
            plots = self.plots_library[plot_ids]
            plots(
                data=data,
                out_dir=self.out_dir)

    def run_workflow(self, workflow_id):
        workflow = self.workflows_library[workflow_id]
        experiment_id = workflow['experiment']
        plot_ids = workflow['plots']

        # run the experiment
        self.output_data = self.run_experiment(experiment_id)

        # run the plots with the data
        self.run_plots(plot_ids, self.output_data)



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
    control.run_workflow('1')


if __name__ == '__main__':
    toy_control()
