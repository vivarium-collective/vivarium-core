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


DEFAULT_PROCESS_SLEEP = 1e-3
DEFAULT_N_PROCESSES = 10
DEFAULT_N_VARIABLES = 10
DEFAULT_EXPERIMENT_TIME = 100


PROCESS_UPDATE_MARKER = 'b.'
VIVARIUM_OVERHEAD_MARKER = 'r.'
SIMULATION_TIME_MARKER = 'g.'


class ManyVariablesProcess(Process):
    defaults = {
        'number_of_ports': 1,
        'number_of_variables': DEFAULT_N_VARIABLES,
        'process_sleep': DEFAULT_PROCESS_SLEEP,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        # make a bunch of port and variable ids
        port_ids = [*range(
            self.parameters['number_of_ports'])]
        variable_ids = [*range(
            self.parameters['number_of_variables'])]

        # assign variables to each port
        self.port_variables = {port_id: [] for port_id in port_ids}
        port_idx = 0
        for var_id in variable_ids:
            port_id = port_ids[port_idx]
            self.port_variables[port_id].append(var_id)
            if port_idx >= (self.parameters['number_of_ports'] - 1):
                port_idx = 0
            else:
                port_idx += 1

    def ports_schema(self):
        ports = {
            port_id: {
                variable: {
                    '_default': random.random(),
                    '_emit': True
                } for variable in variable_ids
            } for port_id, variable_ids in self.port_variables.items()}
        return ports

    def next_update(self, timestep, states):
        update = {}
        for port_id, variable_ids in self.port_variables.items():
            port_update = {
                variable: random.random()
                for variable in variable_ids}
            update[port_id] = port_update

        time.sleep(self.parameters['process_sleep'])
        return update


class ManyVariablesComposite(Composer):
    defaults = {
        'number_of_processes': DEFAULT_N_PROCESSES,
        'number_of_variables': DEFAULT_N_VARIABLES,
        'process_sleep': DEFAULT_PROCESS_SLEEP,
        'number_of_parallel_processes': 0,
        'number_of_stores': 10,
        'number_of_ports': 1,
        'hierarchy_depth': 1,
    }

    def __init__(self, config=None):
        super().__init__(config)
        process_ids = [
            f'process_{key}' for key
            in range(self.config['number_of_processes'])]
        parallel_process_ids = [
            process_id for process_idx, process_id in enumerate(process_ids, 1)
            if process_idx <= self.config['number_of_parallel_processes']]
        store_ids = [
            f'store_{key}' for key
            in range(self.config['number_of_stores'])]

        # make a bunch of processes
        self.processes = {
            process_id: ManyVariablesProcess({
                'name': process_id,
                'number_of_ports': self.config['number_of_ports'],
                'number_of_variables': self.config['number_of_variables'],
                'process_sleep': self.config['process_sleep'],
                '_parallel': process_id in parallel_process_ids,
            })
            for process_id in process_ids}

        # connect the processes' ports to a random store at hierarchy_depth
        self.topology = {}
        for process_id, process in self.processes.items():
            process_ports = {}
            for port_id in process.port_variables.keys():
                store_id = random.choice(store_ids)
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
    number_of_processes = DEFAULT_N_PROCESSES
    number_of_variables = DEFAULT_N_VARIABLES
    process_sleep = DEFAULT_PROCESS_SLEEP
    number_of_parallel_processes = 0
    number_of_stores = 10
    number_of_ports = 1
    hierarchy_depth = 1
    experiment_time = DEFAULT_EXPERIMENT_TIME

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
            number_of_parallel_processes=None,
            number_of_stores=None,
            number_of_ports=None,
            number_of_variables=None,
            hierarchy_depth=None,
            process_sleep=None,
            print_top_stats=None,
            experiment_time=None,
    ):
        self.number_of_processes = \
            number_of_processes or self.number_of_processes
        self.number_of_parallel_processes = \
            number_of_parallel_processes or self.number_of_parallel_processes
        self.number_of_ports = \
            number_of_ports or self.number_of_ports
        self.number_of_variables = \
            number_of_variables or self.number_of_variables
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
        number_of_parallel_processes = kwargs.get(
            'number_of_parallel_processes', self.number_of_parallel_processes)
        number_of_stores = kwargs.get(
            'number_of_stores', self.number_of_stores)
        number_of_ports = kwargs.get(
            'number_of_ports', self.number_of_ports)
        number_of_variables = kwargs.get(
            'number_of_variables', self.number_of_variables)
        hierarchy_depth = kwargs.get(
            'hierarchy_depth', self.hierarchy_depth)
        process_sleep = kwargs.get(
            'process_sleep', self.process_sleep)

        composer = ManyVariablesComposite({
            'number_of_processes': number_of_processes,
            'number_of_parallel_processes': number_of_parallel_processes,
            'number_of_stores': number_of_stores,
            'number_of_ports': number_of_ports,
            'number_of_variables': number_of_variables,
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
        self.experiment.end()

    def _get_emitter_data(self, **kwargs):
        _ = kwargs
        data = self.experiment.emitter.get_data()
        return data

    def _get_emitter_timeseries(self, **kwargs):
        _ = kwargs
        timeseries = self.experiment.emitter.get_timeseries()
        return timeseries

    def _profile_method(self, method, **kwargs):
        """The main profiling method and of the simulation steps

        Args
            method: the simulation step. For example self._run_experiment
        """
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

    def profile_communication_latency(self):

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

        return process_update_time, store_update_time


def run_scan(
    sim,
    scan_values=None,
):
    """Run a scan

    Args
        sim: the ComplexModelSim object.
        scan_values: a list of dicts, with individual scan values.
    """
    scan_values = scan_values or []

    saved_stats = []
    for scan_dict in scan_values:
        n_processes = scan_dict.get('number_of_processes', DEFAULT_N_PROCESSES)
        n_vars = scan_dict.get('number_of_variables', DEFAULT_N_VARIABLES)
        n_parallel_processes = scan_dict.get('number_of_parallel_processes', 0)
        n_stores = scan_dict.get('number_of_stores', 10)
        n_ports = scan_dict.get('number_of_ports', 1)
        hierarchy_depth = scan_dict.get('hierarchy_depth', 1)

        # set the parameters
        sim.set_parameters(
            number_of_processes=n_processes,
            number_of_parallel_processes=n_parallel_processes,
            number_of_stores=n_stores,
            number_of_ports=n_ports,
            number_of_variables=n_vars,
            hierarchy_depth=hierarchy_depth,
        )

        print(
            f'number_of_processes={n_processes}, '
            f'number_of_stores={n_stores}, '
            f'number_of_ports={n_ports}, '
            f'number_of_variables={n_vars}, '
            f'hierarchy_depth={hierarchy_depth}, '
            f'number_of_parallel_processes={n_parallel_processes} '
        )

        # run experiment
        process_update_time, store_update_time = \
            sim.profile_communication_latency()

        # save data
        stat_dict = {
            'number_of_processes': n_processes,
            'number_of_stores': n_stores,
            'number_of_ports': n_ports,
            'number_of_variables': n_vars,
            'hierarchy_depth': hierarchy_depth,
            'number_of_parallel_processes': n_parallel_processes,
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


def get_patches(
        process=True,
        overhead=True,
        experiment=False
):
    patches = []
    if process:
        patches.append(mpatches.Patch(
            color=PROCESS_UPDATE_MARKER[0], label="process updates"))
    if overhead:
        patches.append(mpatches.Patch(
            color=VIVARIUM_OVERHEAD_MARKER[0], label="vivarium overhead"))
    if experiment:
        patches.append(mpatches.Patch(
            color=SIMULATION_TIME_MARKER[0], label="simulation time"))
    return patches


def plot_scan_results(
        saved_stats,
        plot_all=True,
        process_plot=False,
        store_plot=False,
        port_plot=False,
        var_plot=False,
        hierarchy_plot=False,
        parallel_plot=False,
        out_dir='out/experiments',
        filename='profile',
):
    if plot_all:
        process_plot = True
        store_plot = True
        port_plot = True
        var_plot = True
        hierarchy_plot = True
        parallel_plot = True

    n_cols = 1
    n_rows = sum([
        process_plot,
        store_plot,
        port_plot,
        var_plot,
        hierarchy_plot,
        parallel_plot,
    ])

    # make figure
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
            label='number of variables')
        plot_n += 1

    if hierarchy_plot:
        ax_depth = make_axis(
            fig, grid, plot_n, patches,
            label='hierarchy depth')
        plot_n += 1

    if parallel_plot:
        patches = get_patches(
            process=False,
            overhead=False,
            experiment=True)
        ax_depth = make_axis(
            fig, grid, plot_n, patches,
            label='number of parallel processes')
        plot_n += 1

    # plot saved states
    for stat in saved_stats:
        n_processes = stat['number_of_processes']
        n_stores = stat['number_of_stores']
        n_ports = stat['number_of_ports']
        n_vars = stat['number_of_variables']
        n_parallel_processes = stat['number_of_parallel_processes']
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

        if parallel_plot:
            experiment_time = process_update_time + store_update_time
            ax_depth.plot(
                n_parallel_processes, experiment_time, SIMULATION_TIME_MARKER)

    # adjustments
    plt.subplots_adjust(hspace=0.5)
    plt.figtext(0, -0.1, filename, size=8)

    # save
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, filename[0:100])
    fig.savefig(fig_path, bbox_inches='tight')


# scan functions
def scan_stores():
    n_stores = [n*100 for n in range(10)]
    scan_values = [{
        'number_of_stores': n} for n in n_stores]

    sim = ComplexModelSim()
    sim.experiment_time = 100
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      store_plot=True,
                      var_plot=True,
                      filename='scan_stores')


def scan_processes():
    n_processes = [n*40 for n in range(10)]
    scan_values = [{'number_of_processes': n} for n in n_processes]

    sim = ComplexModelSim()
    sim.experiment_time = 100
    sim.process_sleep = 1e-4
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      process_plot=True,
                      filename='scan_processes')


def scan_variables():
    n_vars = [n*100 for n in range(10)]
    scan_values = [{'number_of_variables': n} for n in n_vars]

    sim = ComplexModelSim()
    sim.experiment_time = 5
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      var_plot=True,
                      filename='scan_variables')


def scan_number_of_ports():
    n_ports = [n*10 for n in range(10)]
    scan_values = [
        {
            'number_of_processes': 3,
            'number_of_stores': 10,
            'number_of_variables': 100,
            'number_of_ports': n
        } for n in n_ports
    ]

    sim = ComplexModelSim()
    sim.experiment_time = 100
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      port_plot=True,
                      filename='scan_number_of_ports')


def scan_hierarchy_depth():
    hierarchy_depth = [n*2 for n in range(20)]
    scan_values = [
        {
            'number_of_stores': 10,
            'number_of_variables': 5,
            'number_of_ports': 1,
            'hierarchy_depth': n,
        } for n in hierarchy_depth
    ]

    sim = ComplexModelSim()
    sim.experiment_time = 100
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      hierarchy_plot=True,
                      filename='scan_hierarchy_depth')


def scan_parallel_processes():
    total_processes = 20
    n_parallel_processes = [i*2 for i in range(int(total_processes/2))]
    scan_values = [
        {
            'number_of_processes': total_processes,
            'number_of_parallel_processes': n
        } for n in n_parallel_processes
    ]

    sim = ComplexModelSim()
    sim.experiment_time = 100
    sim.process_sleep = 1e-2
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      plot_all=False,
                      parallel_plot=True,
                      filename=f'scan_parallel_processes_{total_processes}')


scans_library = {
    '0': scan_stores,
    '1': scan_processes,
    '2': scan_variables,
    '3': scan_number_of_ports,
    '4': scan_hierarchy_depth,
    '5': scan_parallel_processes,
}

# python vivarium/experiments/profile_runtime.py -n [name]
if __name__ == '__main__':
    run_library_cli(scans_library)
