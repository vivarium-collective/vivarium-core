"""
===================
Experiment Control
===================

Run experiments and analyses from the command line
"""

import os
import argparse
import copy

from vivarium.core.experiment import timestamp
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    ToyCompartment,
    BASE_OUT_DIR,
)

from vivarium.plots.simulation_output import plot_simulation_output



def make_dir(out_dir):
    os.makedirs(out_dir, exist_ok=True)


class Control(object):

    def __init__(
            self,
            out_dir=None,
            experiments=None,
            compartments=None,
            plots=None,
            workflows=None,
    ):
        if workflows is None:
            workflows = {}
        if experiments is None:
            experiments = {}

        self.experiments_library = experiments
        self.compartments_library = compartments
        self.plots_library = plots
        self.workflows_library = workflows
        self.output_data = None

        # base output directory
        self.out_dir = BASE_OUT_DIR
        if out_dir:
            self.out_dir = out_dir

        # arguments
        self.args = self.add_arguments()

        if self.args.experiment:
            experiment_id = str(self.args.experiment)
            self.output_data = self.run_experiment(experiment_id)

        if self.args.workflow:
            workflow_id = str(self.args.workflow)
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

    def run_one_plot(self, plot_id, data, out_dir=None):
        data_copy = copy.deepcopy(data)
        plot_spec = self.plots_library[plot_id]
        if isinstance(plot_spec, dict):
            # retrieve plot and config from dictionary
            config = plot_spec.get('config', {})
            plot = plot_spec['plot']
            plot(
                data=data_copy,
                config=config,
                out_dir=out_dir)

        elif callable(plot_spec):
            # call plot directly
            plot_spec(
                data=data_copy,
                out_dir=out_dir)

    def run_plots(self, plot_ids, data, out_dir=None):
        if out_dir is None:
            out_dir = self.out_dir
        make_dir(out_dir)

        if isinstance(plot_ids, list):
            for plot_id in plot_ids:
                self.run_one_plot(plot_id, data, out_dir=out_dir)
        else:
            self.run_one_plot(plot_ids, data, out_dir=out_dir)

    def run_workflow(self, workflow_id):
        workflow = self.workflows_library[workflow_id]
        experiment_id = workflow['experiment']
        plot_ids = workflow['plots']

        # output directory for this workflow
        workflow_name = workflow.get('name', timestamp())
        out_dir = os.path.join(self.out_dir, workflow_name)

        # run the experiment
        self.output_data = self.run_experiment(experiment_id)

        # run the plots
        self.run_plots(plot_ids, self.output_data, out_dir=out_dir)

        print('plots saved to directory: {}'.format(out_dir))



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


def toy_plot(data, config=None, out_dir='out'):
    plot_simulation_output(data, out_dir=out_dir)


def toy_control():
    """ a toy example of control

    To run:
    > python vivarium/core/control.py -w 1
    """
    experiment_library = {
        # put in dictionary with name
        '1': {
            'name': 'exp_1',
            'experiment': toy_experiment},
        # map to function to run as is
        '2': toy_experiment,
    }
    plot_library = {
        # put in dictionary with config
        '1': {
            'plot': toy_plot,
            'config': {}},
        # map to function to run as is
        '2': toy_plot
    }
    compartment_library = {
        'agent': ToyCompartment,
    }
    workflow_library = {
        '1': {
            'name': 'test_workflow',
            'experiment': '1',
            'plots': ['1']}
    }

    control = Control(
        out_dir=os.path.join('out', 'control_test'),
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
