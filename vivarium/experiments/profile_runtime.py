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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from vivarium.core.engine import Engine
from vivarium.core.process import Process
from vivarium.core.composer import Composer
from vivarium.core.control import run_library_cli
from vivarium.core.composition import EXPERIMENT_OUT_DIR

DEFAULT_PROCESS_SLEEP = 1e-3
DEFAULT_N_PROCESSES = 10
DEFAULT_N_VARIABLES = 10
DEFAULT_EXPERIMENT_TIME = 100
PROCESS_UPDATE_MARKER = 'bD'
VIVARIUM_OVERHEAD_MARKER = 'rD'
SIMULATION_TIME_MARKER = 'gD'


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

        # use while loop for busy wait, to use CPU time
        current_time = time.time()
        while time.time() < current_time + self.parameters['process_sleep']:
            pass

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
        # TODO -- control number of stores at hierarchy depth
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

        process_update_time = 0
        for s in stats_list:
            process_update_time += stats.stats[s][3]

        # get runtime
        experiment_time = stats.total_tt
        store_update_time = experiment_time - process_update_time

        return process_update_time, store_update_time


# Parameter scan functions
##########################

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
        # set the parameters
        sim.set_parameters(
            number_of_processes=scan_dict.get('number_of_processes'),
            process_sleep=scan_dict.get('process_sleep'),
            number_of_parallel_processes=scan_dict.get(
                'number_of_parallel_processes'),
            number_of_stores=scan_dict.get('number_of_stores'),
            number_of_ports=scan_dict.get('number_of_ports'),
            number_of_variables=scan_dict.get('number_of_variables'),
            hierarchy_depth=scan_dict.get('hierarchy_depth'),
        )

        print(f"{scan_dict}")

        # run experiment
        process_update_time, store_update_time = \
            sim.profile_communication_latency()

        stat_dict = {
            **scan_dict,
            'process_update_time': process_update_time,
            'store_update_time': store_update_time,
        }
        saved_stats.append(stat_dict)

    return saved_stats


# Plotting functions
####################

def _make_axis(fig, grid, plot_n, patches, title='', label=''):
    ax = fig.add_subplot(grid[plot_n, 0])
    ax.set_xlabel(label)
    ax.set_ylabel('wall time (s)')
    ax.set_title(title)
    ax.legend(
        handles=patches,
        # ncol=2,
        # bbox_to_anchor=(0.5, 1.2),  # above
        bbox_to_anchor=(1.45, 0.65),  # to the right
    )
    return ax


def _get_patches(
        process=False,
        overhead=False,
        experiment=False
):
    patches = []
    if process:
        patches.append(mpatches.Patch(
            color=PROCESS_UPDATE_MARKER[0],
            label="process updates"))
    if overhead:
        patches.append(mpatches.Patch(
            color=VIVARIUM_OVERHEAD_MARKER[0],
            label="vivarium overhead"))
    if experiment:
        patches.append(mpatches.Patch(
            color=SIMULATION_TIME_MARKER[0],
            label="simulation time"))
    return patches


def _add_stats_plot(
        ax,
        saved_stats,
        variable_name,
        process_update=False,
        vivarium_overhead=False,
        experiment_time=False,
        markersize=10,
):
    # plot saved states
    for stat in saved_stats:
        variable = stat[variable_name]
        process_update_time = stat['process_update_time']
        store_update_time = stat['store_update_time']

        if process_update:
            ax.plot(
                variable, process_update_time,
                PROCESS_UPDATE_MARKER, markersize=markersize)
        if vivarium_overhead:
            ax.plot(
                variable, store_update_time,
                VIVARIUM_OVERHEAD_MARKER, markersize=markersize)
        if experiment_time:
            experiment_time = process_update_time + store_update_time
            ax.plot(
                variable, experiment_time,
                SIMULATION_TIME_MARKER, markersize=markersize)


def plot_scan_results(
        saved_stats,
        process_plot=False,
        store_plot=False,
        port_plot=False,
        var_plot=False,
        hierarchy_plot=False,
        parallel_plot=False,
        cpus_plot=False,
        fig=None,
        grid=None,
        axis_number=0,
        row_height=3,
        title=None,
        out_dir=EXPERIMENT_OUT_DIR,
        filename=None,
):
    """Plot scan results

    Args
        saved_stats (dict): the scan results
        *_plot (bool): whether to add the given plot type
    """
    # make figure
    if fig:
        assert grid, "fig must provide grid for subplots"
    else:
        n_cols = 1
        n_rows = sum([
            process_plot,
            store_plot,
            port_plot, port_plot,  # two rows
            var_plot, var_plot,  # two rows
            hierarchy_plot,
            parallel_plot,
            cpus_plot,
        ])

        fig = plt.figure(figsize=(n_cols * 6, n_rows * row_height))
        grid = plt.GridSpec(n_rows, n_cols)

    patches = _get_patches(
        process=True, overhead=True, experiment=True)

    # initialize axes
    if process_plot:
        ax = _make_axis(
            fig, grid, axis_number, patches, title,
            label='n processes')
        _add_stats_plot(
            ax=ax, saved_stats=saved_stats,
            variable_name='number_of_processes',
            process_update=True,
            vivarium_overhead=True)
        axis_number += 1

    if store_plot:
        ax = _make_axis(
            fig, grid, axis_number, patches, title,
            label='n stores')
        _add_stats_plot(
            ax=ax, saved_stats=saved_stats,
            variable_name='number_of_stores',
            vivarium_overhead=True)
        axis_number += 1

    if port_plot:
        label = 'n ports'
        var_name = 'number_of_ports'
        ax = _make_axis(
            fig, grid, axis_number, patches, title,
            label=label)
        _add_stats_plot(
            ax=ax, saved_stats=saved_stats,
            variable_name=var_name,
            # process_update=True,
            vivarium_overhead=True)
        axis_number += 1

        # process time plot
        ax = _make_axis(
            fig, grid, axis_number, patches, title='',
            label=label)
        _add_stats_plot(
            ax=ax, saved_stats=saved_stats,
            variable_name=var_name,
            process_update=True)
        axis_number += 1

    if var_plot:
        label = 'n variables'
        var_name = 'number_of_variables'
        ax = _make_axis(
            fig, grid, axis_number, patches, title,
            label=label)
        _add_stats_plot(
            ax=ax, saved_stats=saved_stats,
            variable_name=var_name,
            vivarium_overhead=True)
        axis_number += 1

        # process time plot
        ax = _make_axis(
            fig, grid, axis_number, patches, title='',
            label=label)
        _add_stats_plot(
            ax=ax, saved_stats=saved_stats,
            variable_name=var_name,
            process_update=True)
        axis_number += 1

    if hierarchy_plot:
        ax = _make_axis(
            fig, grid, axis_number, patches, title,
            label='hierarchy depth')
        _add_stats_plot(
            ax=ax, saved_stats=saved_stats,
            variable_name='hierarchy_depth',
            vivarium_overhead=True)
        axis_number += 1

    if parallel_plot:
        ax = _make_axis(
            fig, grid, axis_number, patches, title,
            label='n parallel processes')
        _add_stats_plot(
            ax=ax, saved_stats=saved_stats,
            variable_name='number_of_parallel_processes',
            experiment_time=True)
        axis_number += 1

    if cpus_plot:
        ax = _make_axis(
            fig, grid, axis_number, patches, title,
            label='n vCPUs')
        _add_stats_plot(
            ax=ax, saved_stats=saved_stats,
            variable_name='number_of_cpus',
            experiment_time=True)
        axis_number += 1

    # save
    if filename:
        plt.subplots_adjust(hspace=0.5)
        # plt.figtext(0, -0.1, filename, size=8)
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, filename[0:100])
        fig.savefig(fig_path, bbox_inches='tight')
    return fig


# Individual scan functions
###########################

def scan_processes():
    n_processes = [n*15 for n in range(10)]
    sleep_times = [0.8e-5, 0.8e-4, 0.8e-3]

    n_cols = 1
    n_rows = len(sleep_times)
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 3))
    grid = plt.GridSpec(n_rows, n_cols)

    for idx, s in enumerate(sleep_times):
        scan_values = []
        for n in n_processes:
            scan_value = {
                'number_of_processes': n,
                'process_sleep': s,
            }
            scan_values.append(scan_value)

        sim = ComplexModelSim()
        saved_stats = run_scan(sim,
                               scan_values=scan_values)

        plot_scan_results(saved_stats,
                          process_plot=True,
                          row_height=2.5,
                          fig=fig,
                          grid=grid,
                          axis_number=idx,
                          title=f'process update runtime = {s} sec',
                          # filename='scan_processes'
                          )
    plot_scan_results({},
                      fig=fig,
                      grid=grid,
                      filename='scan_processes'
                      )


def scan_variables():
    n_vars = [n*100 for n in range(10)]
    scan_values = [{'number_of_variables': n} for n in n_vars]

    sim = ComplexModelSim()
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      var_plot=True,
                      row_height=1.8,
                      filename='scan_variables',
                      title='n variables through 1 port'
                      )


def scan_number_of_ports():
    n_ports = [n*10 for n in range(10)]
    scan_values = [
        {
            'number_of_variables': 100,
            'number_of_ports': n
        } for n in n_ports
    ]

    sim = ComplexModelSim()
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      port_plot=True,
                      row_height=1.8,
                      filename='scan_number_of_ports',
                      title='100 variables through n ports'
                      )


def scan_hierarchy_depth():
    hierarchy_depth = [n*2 for n in range(20)]
    scan_values = [{'hierarchy_depth': n} for n in hierarchy_depth]

    sim = ComplexModelSim()
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      hierarchy_plot=True,
                      filename='scan_hierarchy_depth')


def scan_parallel_processes():
    """
    Running this scan may require change the computer's process limits.
     - Check your current limits with: `ulimit -n`
     - Increase limits: `ulimit -n 10240`
    """

    total_processes = 80
    scan_interval = 10
    scan_values = [
        {
            'number_of_processes': total_processes,
            'number_of_parallel_processes': n
        } for n in range(0, total_processes+1, scan_interval)
    ]

    sim = ComplexModelSim()
    sim.process_sleep = 1e-2
    saved_stats = run_scan(sim,
                           scan_values=scan_values)
    plot_scan_results(saved_stats,
                      parallel_plot=True,
                      row_height=2.5,
                      title=f'{total_processes} processes, '
                            f'with n of them running in parallel',
                      filename='scan_parallel_processes')


scans_library = {
    '0': scan_processes,
    '1': scan_variables,
    '2': scan_number_of_ports,
    '3': scan_parallel_processes,
    '4': scan_hierarchy_depth,
}

# python vivarium/experiments/profile_runtime.py -n [name]
if __name__ == '__main__':
    run_library_cli(scans_library)
