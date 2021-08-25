"""
Experiment to profile runtime in process next_update vs
remaining vivarium overhead

Execute by running:
``python vivarium/experiments/profile_runtime.py -n [scan id]``
"""

import os
import random
import time
import cProfile
import pstats
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from vivarium.core.engine import Engine
from vivarium.core.process import Process
from vivarium.core.composer import Composer
from vivarium.core.control import run_library_cli


class ManyVariablesProcess(Process):
    defaults = {
        'number_of_ports': 1,
        'variables_per_port': 10,
        'process_sleep': 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        # make a bunch of port and variable ids
        self.port_ids = [*range(
            self.parameters['number_of_ports'])]
        self.variable_ids = [*range(
            self.parameters['variables_per_port'])]

    def ports_schema(self):
        ports = {
            port_id: {
                variable: {
                    '_default': random.random(),
                    '_emit': True
                } for variable in self.variable_ids
            } for port_id in self.port_ids}
        return ports

    def next_update(self, timestep, states):
        update = {}
        for port_id in self.port_ids:
            port_update = {
                variable: random.random()
                for variable in self.variable_ids}
            update[port_id] = port_update

        time.sleep(self.parameters['process_sleep'])
        return update

class ManyVariablesComposite(Composer):
    defaults = {
        'number_of_processes': 10,
        'number_of_stores': 10,
        'number_of_ports': 1,
        'variables_per_process_port': 10,
        'hierarchy_depth': 1,
        'process_sleep': 0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.process_ids = [
            f'process_{key}' for key
            in range(self.config['number_of_processes'])]
        self.store_ids = [
            f'store_{key}' for key
            in range(self.config['number_of_stores'])]

        # make a bunch of processes
        self.processes = {
            process_id: ManyVariablesProcess({
                'name': process_id,
                'number_of_ports': self.config['number_of_ports'],
                'variables_per_port': self.config['variables_per_process_port'],
                'process_sleep': self.config['process_sleep'],
            })
            for process_id in self.process_ids}

        # connect the processes' ports to a random store at hierarchy_depth
        self.topology = {}
        for process_id, process in self.processes.items():
            process_ports = {}
            for port_id in process.port_ids:
                store_id = random.choice(self.store_ids)
                store_path = [
                    store_id for _
                    in range(self.config['hierarchy_depth'])]
                process_ports[port_id] = tuple(store_path)
            self.topology[process_id] = process_ports

    def generate_processes(self, config):
        return self.processes

    def generate_topology(self, config):
        return self.topology


class ComplexModelSim:
    """Profile Complex Models

    This class lets you initialize and profile the simulation of
    composite models with arbitrary numbers of processes, variables
    per process, and total stores.
    """

    # model complexity
    number_of_processes = 10
    number_of_stores = 10
    number_of_ports = 1
    variables_per_port = 10
    hierarchy_depth = 1
    process_sleep = 0
    experiment_time = 100

    # display
    print_top_stats = 4

    # initialize
    composite = None
    experiment = None

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
            number_of_stores=None,
            number_of_ports=None,
            variables_per_port=None,
            hierarchy_depth=None,
            process_sleep=None,
            print_top_stats=None,
            experiment_time=None,
    ):
        self.number_of_processes = \
            number_of_processes or self.number_of_processes
        self.number_of_ports = \
            number_of_ports or self.number_of_ports
        self.variables_per_port = \
            variables_per_port or self.variables_per_port
        self.number_of_stores = \
            number_of_stores or self.number_of_stores
        self.hierarchy_depth = \
            hierarchy_depth or self.hierarchy_depth
        self.process_sleep = \
            process_sleep or self.process_sleep
        self.print_top_stats = \
            print_top_stats or self.print_top_stats
        self.experiment_time = \
            experiment_time or self.experiment_time

    def _generate_composite(self, **kwargs):
        number_of_processes = kwargs.get(
            'number_of_processes', self.number_of_processes)
        number_of_stores = kwargs.get(
            'number_of_stores', self.number_of_stores)
        number_of_ports = kwargs.get(
            'number_of_ports', self.number_of_ports)
        variables_per_port = kwargs.get(
            'variables_per_port', self.variables_per_port)
        hierarchy_depth = kwargs.get(
            'hierarchy_depth', self.hierarchy_depth)
        process_sleep = kwargs.get(
            'process_sleep', self.process_sleep)

        composer = ManyVariablesComposite({
            'number_of_processes': number_of_processes,
            'number_of_stores': number_of_stores,
            'number_of_ports': number_of_ports,
            'variables_per_port': variables_per_port,
            'hierarchy_depth': hierarchy_depth,
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
        _ = kwargs
        data = self.experiment.emitter.get_data()
        return data

    def _get_emitter_timeseries(self, **kwargs):
        _ = kwargs
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
        _, stats_list = stats.get_print_list(next_update_amount)
        cc, nc, tt, ct, callers = stats.stats[stats_list[0]]
        _ = cc
        _ = nc
        _ = tt
        _ = callers
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
    sim=None,
    scan_values=None,
):
    sim = sim or ComplexModelSim()
    scan_values = scan_values or [{
        'number_of_stores': 10,
        'number_of_processes': 10,
        'variables_per_port': 10,
        'number_of_ports': 1,
        'hierarchy_depth': 1,
    }]

    saved_stats = []
    for scan_dict in scan_values:
        n_processes = scan_dict.get('number_of_processes', 10)
        n_stores = scan_dict.get('number_of_stores', 10)
        n_vars = scan_dict.get('variables_per_port', n_stores)
        n_ports = scan_dict.get('number_of_ports', 1)
        hierarchy_depth = scan_dict.get('hierarchy_depth', 1)

        # set the parameters
        sim.set_parameters(
            number_of_processes=n_processes,
            number_of_stores=n_stores,
            number_of_ports=n_ports,
            variables_per_port=n_vars,
            hierarchy_depth=hierarchy_depth,
        )

        print(
            f'number_of_processes={n_processes}, '
            f'number_of_stores={n_stores}, '
            f'number_of_ports={n_ports}, '
            f'variables_per_port={n_vars}, '
            f'hierarchy_depth={hierarchy_depth}, '
        )

        # run experiment
        process_update_time, store_update_time = \
            sim.profile_communication_latency(print_report=False)

        # save data
        stat_dict = {
            'number_of_processes': n_processes,
            'number_of_stores': n_stores,
            'number_of_ports': n_ports,
            'variables_per_port': n_vars,
            'hierarchy_depth': hierarchy_depth,
            'process_update_time': process_update_time,
            'store_update_time': store_update_time,
        }
        saved_stats.append(stat_dict)

    return saved_stats


def make_axis(fig, grid, plot_n, patches, label=''):
    ax = fig.add_subplot(grid[plot_n, 0])
    ax.set_xlabel(label)
    ax.set_ylabel('runtime (s)')
    ax.legend(
        loc='upper center',
        handles=patches, ncol=2,
        bbox_to_anchor=(0.5, 1.2), )
    return ax

PROCESS_UPDATE_MARKER = 'b.'
VIVARIUM_OVERHEAD_MARKER = 'r.'

def get_patches():
    patches = []
    patches.append(
        mpatches.Patch(
            color=PROCESS_UPDATE_MARKER[0],
            label="process updates"))
    patches.append(
        mpatches.Patch(
            color=VIVARIUM_OVERHEAD_MARKER[0],
            label="vivarium overhead"))
    return patches



def plot_scan_results(
        saved_stats,
        plot_all=True,
        process_plot=False,
        store_plot=False,
        port_plot=False,
        var_plot=False,
        hierarchy_plot=False,
        out_dir='out/experiments',
        filename='profile',
):
    if plot_all:
        process_plot = True
        store_plot = True
        port_plot = True
        var_plot = True
        hierarchy_plot = True

    n_cols = 1
    n_rows = sum([
        process_plot,
        store_plot,
        port_plot,
        var_plot,
        hierarchy_plot])

    ## make figure
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 3))
    grid = plt.GridSpec(n_rows, n_cols)

    patches = get_patches()

    # initialize axes
    plot_n = 0
    if process_plot:
        ax_nprocesses = make_axis(
            fig, grid, plot_n, patches,
            label='number of processes')
        plot_n += 1

    if store_plot:
        ax_nstores = make_axis(
            fig, grid, plot_n, patches,
            label='number of stores')
        plot_n += 1

    if port_plot:
        ax_nports = make_axis(
            fig, grid, plot_n, patches,
            label='number of ports')
        plot_n += 1

    if var_plot:
        ax_nvars = make_axis(
            fig, grid, plot_n, patches,
            label='number of variables per port')
        plot_n += 1

    if hierarchy_plot:
        ax_depth = make_axis(
            fig, grid, plot_n, patches,
            label='hierarchy depth')
        plot_n += 1

    # plot saved states
    for stat in saved_stats:
        n_processes = stat['number_of_processes']
        n_stores = stat['number_of_stores']
        n_ports = stat['number_of_ports']
        n_vars = stat['variables_per_port']
        depth = stat['hierarchy_depth']
        process_update_time = stat['process_update_time']
        store_update_time = stat['store_update_time']

        if process_plot:
            ax_nprocesses.plot(
                n_processes, process_update_time, PROCESS_UPDATE_MARKER)
            ax_nprocesses.plot(
                n_processes, store_update_time, VIVARIUM_OVERHEAD_MARKER)

        if store_plot:
            ax_nstores.plot(
                n_stores, process_update_time, PROCESS_UPDATE_MARKER)
            ax_nstores.plot(
                n_stores, store_update_time, VIVARIUM_OVERHEAD_MARKER)

        if port_plot:
            ax_nports.plot(
                n_ports, process_update_time, PROCESS_UPDATE_MARKER)
            ax_nports.plot(
                n_ports, store_update_time, VIVARIUM_OVERHEAD_MARKER)

        if var_plot:
            ax_nvars.plot(
                n_vars, process_update_time, PROCESS_UPDATE_MARKER)
            ax_nvars.plot(
                n_vars, store_update_time, VIVARIUM_OVERHEAD_MARKER)

        if hierarchy_plot:
            ax_depth.plot(
                depth, process_update_time, PROCESS_UPDATE_MARKER)
            ax_depth.plot(
                depth, store_update_time, VIVARIUM_OVERHEAD_MARKER)

    # adjustments
    plt.subplots_adjust(hspace=0.5)

    # save
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, filename[0:100])
    fig.savefig(fig_path, bbox_inches='tight')


# scan functions
def scan_stores():
    n_stores = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    scan_values = [{
        'number_of_stores': n} for n in n_stores]

    sim = ComplexModelSim()
    sim.experiment_time = 100
    sim.process_sleep = 1e-4
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      store_plot=True,
                      var_plot=True,
                      filename='scan_stores')

def scan_processes():
    n_processes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    scan_values = [{'number_of_processes': n} for n in n_processes]

    sim = ComplexModelSim()
    sim.experiment_time = 100
    sim.process_sleep = 1e-5
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      process_plot=True,
                      filename='scan_processes')


def scan_processes_variables():
    n_processes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    n_vars = [5, 100, 1000]
    scan_values = []
    for n_p in n_processes:
        for n_v in n_vars:
            scan = {
                'number_of_processes': n_p,
                'variables_per_port': n_v,
                'number_of_stores': max(n_vars),
            }
            scan_values.append(scan)

    sim = ComplexModelSim()
    sim.experiment_time = 100
    sim.process_sleep = 1e-4
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      process_plot=True,
                      var_plot=True,
                      filename='scan_processes_variables')

def scan_number_of_ports():
    n_ports = [1, 2, 4, 8, 16, 32, 64]
    scan_values = [
        {
            'number_of_processes': 10,
            'number_of_stores': 10,
            'variables_per_port': 5,
            'number_of_ports': n
        } for n in n_ports
    ]

    sim = ComplexModelSim()
    sim.experiment_time = 100
    # sim.process_sleep = 1e-6
    sim.process_sleep = 1e-4
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      port_plot=True,
                      filename='scan_number_of_ports')


def scan_hierarchy_depth():
    hierarchy_depth = [1, 2, 4, 8, 16]
    scan_values = [
        {
            'number_of_processes': 10,
            'number_of_stores': 10,
            'variables_per_port': 5,
            'number_of_ports': 1,
            'hierarchy_depth': n,
        } for n in hierarchy_depth
    ]

    sim = ComplexModelSim()
    sim.experiment_time = 100
    sim.process_sleep = 1e-4
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      hierarchy_plot=True,
                      filename='scan_hierarchy_depth')


scans_library = {
    '0': scan_stores,
    '1': scan_processes,
    '2': scan_processes_variables,
    '3': scan_number_of_ports,
    '4': scan_hierarchy_depth,
}

if __name__ == '__main__':
    run_library_cli(scans_library)
