"""
Experiment to profile runtime in process next_update vs in store
"""
import os
import random
import time
import cProfile
import pstats
import argparse
import matplotlib.pyplot as plt

from vivarium.core.engine import Engine
from vivarium.core.process import Process
from vivarium.core.composer import Composer


class ManyVariablesProcess(Process):
    defaults = {
        'number_of_variables': 10,
        'process_sleep': 0,
    }
    def __init__(self, parameters=None):
        super().__init__(parameters)
        # make a bunch of variables
        random_variables = [
            key for key in range(self.parameters['number_of_variables'])]
        self.parameters['variables'] = random_variables

    def ports_schema(self):
        return {
            'update_timer': {
                '_default': 0,
                '_emit': True,
            },
            'port': {
                variable: {
                    '_default': random.random(),
                    '_emit': True
                } for variable in self.parameters['variables']}}

    def next_update(self, timestep, states):
        update = {
                variable: random.random()
                for variable in self.parameters['variables']}
        time.sleep(self.parameters['process_sleep'])
        return {
            'port': update,
        }

class ManyVariablesComposite(Composer):
    defaults = {
        'number_of_processes': 10,
        'number_of_stores': 10,
        'variables_per_process': 10,
        'process_sleep': 0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.process_ids = [
            f'process_{key}'
            for key in range(self.config['number_of_processes'])]
        self.store_ids = [
            f'store_{key}'
            for key in range(self.config['number_of_stores'])]

    def generate_processes(self, config):
        # make a bunch of processes
        return {
            process_id: ManyVariablesProcess({
                'name': process_id,
                'number_of_variables': self.config['variables_per_process'],
                'process_sleep': self.config['process_sleep'],
            })
            for process_id in self.process_ids}

    def generate_topology(self, config):
        return {
            process_id: {
                'port': (random.choice(self.store_ids),),
            }
            for process_id in self.process_ids}


class ComplexModelSim:
    """Profile Complex Models

    This class lets you initialize and profile the simulation of
    composite models with arbitrary numbers of processes, variables
    per process, and total stores.
    """

    # model complexity
    number_of_processes = 10
    number_of_variables = 10
    number_of_stores = 10
    process_sleep = 1e-4
    experiment_time = 100

    # display
    print_top_stats = 4

    def from_cli(self):
        parser = argparse.ArgumentParser(
            description='complex model simulations with runtime profiling'
        )
        parser.add_argument(
            '--profile', '-p', action="store_true",
            help="run profile of model composition and simulation"
        )
        parser.add_argument(
            '--latency', '-l', action="store_true",
            help="run profile of communication latency in an experiment"
        )
        parser.add_argument(
            '--scan', '-s', action="store_true",
            help="run scan of communication latency"
        )
        args = parser.parse_args()

        if args.profile:
            self.run_profile()
        if args.latency:
            self.profile_communication_latency()
        if args.scan:
            self.run_scan_and_plot()

    def set_parameters(
            self,
            number_of_processes=None,
            number_of_variables=None,
            number_of_stores=None,
            process_sleep=None,
            print_top_stats=None,
            experiment_time=None,
    ):
        self.number_of_processes = \
            number_of_processes or self.number_of_processes
        self.number_of_variables = \
            number_of_variables or self.number_of_variables
        self.number_of_stores = \
            number_of_stores or self.number_of_stores
        self.process_sleep = \
            process_sleep or self.process_sleep
        self.print_top_stats = \
            print_top_stats or self.print_top_stats
        self.experiment_time = \
            experiment_time or self.experiment_time

    def _generate_composite(self, **kwargs):
        number_of_processes = kwargs.get(
            'number_of_processes', self.number_of_processes)
        number_of_variables = kwargs.get(
            'number_of_variables', self.number_of_variables)
        number_of_stores = kwargs.get(
            'number_of_stores', self.number_of_stores)
        process_sleep = kwargs.get(
            'process_sleep', self.process_sleep)

        composer = ManyVariablesComposite({
            'number_of_processes': number_of_processes,
            'number_of_variables': number_of_variables,
            'number_of_stores': number_of_stores,
            'process_sleep': process_sleep,
        })

        self.composite = composer.generate(**kwargs)

    def _initialize_experiment(self, **kwargs):
        self.experiment = Engine(
            processes=self.composite['processes'],
            topology=self.composite['topology'],
            **kwargs)

    def _run_experiment(self, **kwargs):
        self.experiment.update(kwargs['experiment_time'])

    def _get_emitter_data(self, **kwargs):
        data = self.experiment.emitter.get_data()
        return data

    def _get_emitter_timeseries(self, **kwargs):
        timeseries = self.experiment.emitter.get_timeseries()
        return timeseries

    def _profile_method(self, method, **kwargs):
        print_top_stats = kwargs.get(
            'print_top_stats', self.print_top_stats)
        profiler = cProfile.Profile()
        profiler.enable()
        method(**kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler)
        if print_top_stats:
            stats.sort_stats('tottime').print_stats(print_top_stats)
        return stats

    def run_profile(self):

        print('GENERATE COMPOSITE')
        self._profile_method(
            self._generate_composite)

        print('INITIALIZE EXPERIMENT')
        self._profile_method(
            self._initialize_experiment)

        print('RUN EXPERIMENT')
        self._profile_method(
            self._run_experiment, experiment_time=self.experiment_time)

        print('GET EMITTER DATA')
        self._profile_method(
            self._get_emitter_data)

    def profile_communication_latency(self, print_report=True):

        self._generate_composite()
        self._initialize_experiment(display_info=False)

        # profile the experiment
        stats = self._profile_method(
            self._run_experiment,
            experiment_time=self.experiment_time,
            print_top_stats=None)

        # get next_update runtime
        next_update_amount = ("next_update",)
        width, list = stats.get_print_list(next_update_amount)
        cc, nc, tt, ct, callers = stats.stats[list[0]]
        process_update_time = ct

        # get total runtime
        experiment_time = stats.total_tt

        # # print stats
        # stats.print_stats("large_experiment")
        # stats.print_stats("next_update")

        # analyze
        store_update_time = experiment_time - process_update_time

        if print_report:
            print(f"TOTAL EXPERIMENT TIME: {experiment_time}")
            print(f"PROCESS NEXT_UPDATE: {process_update_time}")
            print(f"STORE UPDATE: {store_update_time}")

        return process_update_time, store_update_time

    def run_scan(
        self,
        processes_range=None,
        stores_range=None,
    ):
        processes_range = processes_range or [1, 10]
        stores_range = stores_range or [1, 10]

        sim = ComplexModelSim()

        saved_stats = {}
        for n_processes in processes_range:
            for n_stores in stores_range:

                # set the parameters
                sim.set_parameters(
                    number_of_processes=n_processes,
                    number_of_variables=n_stores,
                    number_of_stores=n_stores)

                # run experiment
                process_update_time, store_update_time = \
                    sim.profile_communication_latency(print_report=False)

                # save data
                saved_stats[(n_processes, n_stores)] = (
                    process_update_time, store_update_time)

        return saved_stats

    def plot_scan_results(
            self,
            saved_stats,
            out_dir='out/experiments',
            filename='profile',
    ):
        n_cols = 1
        n_rows = 2
        column_width = 6
        row_height = 3
        h_space = 0.5

        # make figure and plot
        fig = plt.figure(figsize=(n_cols * column_width, n_rows * row_height))
        grid = plt.GridSpec(n_rows, n_cols)

        # plot
        ax_processes = fig.add_subplot(grid[0, 0])
        ax_stores = fig.add_subplot(grid[1, 0])
        for (n_processes, n_stores), \
            (process_update_time, store_update_time) \
                in saved_stats.items():

            # plot process run tim
            pr_pr_handle, = ax_processes.plot(
                n_processes,
                process_update_time,
                'bo')
            pr_st_handle, = ax_processes.plot(
                n_processes,
                store_update_time,
                'r+')

            # plot store run time
            st_pr_handle, = ax_stores.plot(
                n_stores,
                process_update_time,
                'bo')
            st_st_handle, = ax_stores.plot(
                n_stores,
                store_update_time,
                'r+')

        # axis labels
        # ax_processes.set_title('process updates')
        ax_processes.set_xlabel('number of processes')
        ax_processes.set_ylabel('runtime (s)')
        ax_processes.legend(
            [pr_pr_handle, pr_st_handle],
            ['process update', 'store update'],
            bbox_to_anchor=(1.05, 1))

        # ax_stores.set_title('store updates')
        ax_stores.set_xlabel('number of stores')
        ax_stores.set_ylabel('runtime (s)')
        ax_stores.legend(
            [st_pr_handle, st_st_handle],
            ['process update', 'store update'],
            bbox_to_anchor=(1.05, 1))

        # adjustments
        plt.subplots_adjust(hspace=h_space)

        # save
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, filename)
        fig.savefig(fig_path, bbox_inches='tight')


    def run_scan_and_plot(self):
        saved_stats = self.run_scan(
            processes_range=[10, 100, 500],
            stores_range=[10, 100, 500],
        )
        self.plot_scan_results(saved_stats)



if __name__ == '__main__':
    sim = ComplexModelSim()
    sim.from_cli()
