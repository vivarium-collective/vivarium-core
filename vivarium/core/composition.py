from __future__ import absolute_import, division, print_function

import copy
import csv
import os
import io

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from vivarium.core.emitter import (
    make_path_dict,
    path_timeseries_from_embedded_timeseries,
    path_timeseries_from_data,
)
from vivarium.core.experiment import Experiment
from vivarium.core.process import Process, Deriver, Generator, generate_derivers
from vivarium.core import emitter as emit
from vivarium.library.dict_utils import (
    deep_merge,
    deep_merge_check,
    flatten_timeseries,
    get_path_list_from_dict,
)
from vivarium.library.units import units

from vivarium.processes.timeline import TimelineProcess
from vivarium.processes.nonspatial_environment import NonSpatialEnvironment


REFERENCE_DATA_DIR = os.path.join('vivarium', 'reference_data')
TEST_OUT_DIR = os.path.join('out', 'tests')
PROCESS_OUT_DIR = os.path.join('out', 'processes')
COMPARTMENT_OUT_DIR = os.path.join('out', 'compartments')
EXPERIMENT_OUT_DIR = os.path.join('out', 'experiments')


# loading functions
def make_agents(agent_ids, compartment, config=None):
    if config is None:
        config = {}

    processes = {}
    topology = {}
    for agent_id in agent_ids:
        agent_config = copy.deepcopy(config)
        agent = compartment.generate(dict(
            agent_config,
            agent_id=agent_id))

        # save processes and topology
        processes[agent_id] = agent['processes']
        topology[agent_id] = agent['topology']

    return {
        'processes': processes,
        'topology': topology}


def agent_environment_experiment(
        agents_config=None,
        environment_config=None,
        initial_state=None,
        initial_agent_state=None,
        settings=None,
        invoke=None,
):
    """ Make an experiment with agents placed in an environment under an `agents` store.
     Arguments:
        * **agents_config**: the configuration for the agents
        * **environment_config**: the configuration for the environment
        * **initial_state**: the initial state for the hierarchy, with environment at the
          top level.
        * **initial_agent_state**: the initial_state for agents, set under each agent_id.
        * **settings**: settings include **emitter** and **agent_names**.
        * **invoke**: is the invoke object for calling updates.
    """
    if settings is None:
        settings = {}

    # experiment settings
    emitter = settings.get('emitter', {'type': 'timeseries'})

    # initialize the agents
    if isinstance(agents_config, dict):
        # dict with single agent config
        agent_type = agents_config['type']
        agent_ids = agents_config['ids']
        agent_compartment = agent_type(agents_config['config'])
        agents = make_agents(agent_ids, agent_compartment, agents_config['config'])

        if initial_agent_state:
            initial_state['agents'] = {
                agent_id: initial_agent_state
                for agent_id in agent_ids}

    elif isinstance(agents_config, list):
        # list with multiple agent configurations
        agents = {
            'processes': {},
            'topology': {}}
        for config in agents_config:
            agent_type = config['type']
            agent_ids = config['ids']
            agent_compartment = agent_type(config['config'])
            new_agents = make_agents(agent_ids, agent_compartment, config['config'])
            deep_merge(agents['processes'], new_agents['processes'])
            deep_merge(agents['topology'], new_agents['topology'])

            if initial_agent_state:
                if 'agents' not in initial_state:
                    initial_state['agents'] = {}
                initial_state['agents'].update({
                    agent_id: initial_agent_state
                    for agent_id in agent_ids})

    if 'agents' in initial_state:
        environment_config[
            'config']['diffusion']['agents'] = initial_state['agents']

    # initialize the environment
    environment_type = environment_config['type']
    environment_compartment = environment_type(environment_config['config'])

    # combine processes and topologies
    network = environment_compartment.generate()
    processes = network['processes']
    topology = network['topology']
    processes['agents'] = agents['processes']
    topology['agents'] = agents['topology']

    if settings.get('agent_names') is True:
        # add an AgentNames processes, which saves the current agent names
        # to as store at the top level of the hierarchy
        processes['agent_names'] = AgentNames({})
        topology['agent_names'] = {
            'agents': ('agents',),
            'names': ('names',)
        }

    experiment_config = {
        'processes': processes,
        'topology': topology,
        'emitter': emitter,
        'initial_state': initial_state,
    }
    if settings.get('experiment_name'):
        experiment_config['experiment_name'] = settings.get('experiment_name')
    if settings.get('description'):
        experiment_config['description'] = settings.get('description')
    if invoke:
        experiment_config['invoke'] = invoke
    if 'emit_step' in settings:
        experiment_config['emit_step'] = settings['emit_step']
    return Experiment(experiment_config)

def process_in_compartment(process, topology={}):
    """ put a lone process in a compartment"""
    class ProcessCompartment(Generator):
        def __init__(self, config):
            super(ProcessCompartment, self).__init__(config)
            self.schema_override = {}
            self.topology = topology
            self.process = process(self.config)

        def generate_processes(self, config):
            return {'process': self.process}

        def generate_topology(self, config):
            return {
                'process': {
                    port: self.topology.get(port, (port,)) for port in self.process.ports_schema().keys()}}

    return ProcessCompartment

def make_experiment_from_configs(
    agents_config={},
    environment_config={},
    initial_state={},
    settings={},
):
    # experiment settings
    emitter = settings.get('emitter', {'type': 'timeseries'})

    # initialize the agents
    agent_type = agents_config['agent_type']
    agent_ids = agents_config['agent_ids']
    agent = agent_type(agents_config['config'])
    agents = make_agents(agent_ids, agent, agents_config['config'])

    # initialize the environment
    environment_type = environment_config['environment_type']
    environment = environment_type(environment_config['config'])

    return make_experiment_from_compartments(
        environment.generate({}), agents, emitter, initial_state)


def make_experiment_from_compartment_dicts(
    environment_dict, agents_dict, emitter_dict, initial_state
):
    # environment_dict comes from environment.generate()
    # agents_dict comes from make_agents
    processes = environment_dict['processes']
    topology = environment_dict['topology']
    processes['agents'] = agents_dict['processes']
    topology['agents'] = agents_dict['topology']
    return Experiment({
        'processes': processes,
        'topology': topology,
        'emitter': emitter_dict,
        'initial_state': initial_state})


def process_in_experiment(process, settings={}):
    initial_state = settings.get('initial_state', {})
    emitter = settings.get('emitter', {'type': 'timeseries'})
    emit_step = settings.get('emit_step')
    timeline = settings.get('timeline', [])
    environment = settings.get('environment', {})
    paths = settings.get('topology', {})

    processes = {'process': process}
    topology = {
        'process': {
            port: paths.get(port, (port,)) for port in process.ports_schema().keys()}}


    if timeline:
        # Adding a timeline to a process requires only the timeline
        timeline_process = TimelineProcess(timeline)
        processes.update({'timeline_process': timeline_process})
        topology.update({
            'timeline_process': {
                port: (port,) for port in timeline_process.ports}})

    if environment:
        # Environment requires ports for external, fields, dimensions,
        # and global (for location)
        ports = environment.get(
            'ports',
            {
                'external': ('external',),
                'fields': ('fields',),
                'dimensions': ('dimensions',),
                'global': ('global',),
            }
        )
        environment_process = NonSpatialEnvironment(environment)
        processes.update({'environment_process': environment_process})
        topology.update({
            'environment_process': {
                'external': ports['external'],
                'fields': ports['fields'],
                'dimensions': ports['dimensions'],
                'global': ports['global'],
            },
        })

    # add derivers
    derivers = generate_derivers(processes, topology)
    processes = deep_merge(processes, derivers['processes'])
    topology = deep_merge(topology, derivers['topology'])

    return Experiment({
        'processes': processes,
        'topology': topology,
        'emitter': emitter,
        'emit_step': emit_step,
        'initial_state': initial_state})

def compartment_in_experiment(compartment, settings={}):
    compartment_config = settings.get('compartment', {})
    timeline = settings.get('timeline')
    environment = settings.get('environment')
    outer_path = settings.get('outer_path', tuple())
    emit_step = settings.get('emit_step')

    network = compartment.generate(compartment_config, outer_path)
    processes = network['processes']
    topology = network['topology']

    if timeline is not None:
        # Environment requires ports for all states defined in the timeline, and a global port
        ports = timeline['ports']
        timeline_process = TimelineProcess(timeline)
        processes.update({'timeline_process': timeline_process})
        if 'global' not in ports:
            ports['global'] = ('global',)
        topology.update({
            'timeline_process': {
                port: path
                for port, path in ports.items()}})

    if environment is not None:
        # Environment requires ports for external, fields, dimensions,
        # and global (for location)
        ports = environment.get(
            'ports',
            {
                'external': ('external',),
                'fields': ('fields',),
                'dimensions': ('dimensions',),
                'global': ('global',),
            }
        )
        environment_process = NonSpatialEnvironment(environment)
        processes.update({'environment_process': environment_process})
        topology.update({
            'environment_process': {
                'external': ports['external'],
                'fields': ports['fields'],
                'dimensions': ports['dimensions'],
                'global': ports['global'],
            },
        })

    return Experiment({
        'processes': processes,
        'topology': topology,
        'emitter': settings.get('emitter', {'type': 'timeseries'}),
        'emit_step': emit_step,
        'initial_state': settings.get('initial_state', {})})


# simulation functions
def simulate_process(process, settings={}):
    experiment = process_in_experiment(process, settings)
    return simulate_experiment(experiment, settings)

def simulate_process_in_experiment(process, settings={}):
    experiment = process_in_experiment(process, settings)
    return simulate_experiment(experiment, settings)

def simulate_compartment_in_experiment(compartment, settings={}):
    experiment = compartment_in_experiment(compartment, settings)
    return simulate_experiment(experiment, settings)

def simulate_experiment(experiment, settings={}):
    '''
    run an experiment simulation
        Requires:
        - a configured experiment

    Returns:
        - a timeseries of variables from all ports.
        - if 'return_raw_data' is True, it returns the raw data instead
    '''
    total_time = settings.get('total_time', 10)
    return_raw_data = settings.get('return_raw_data', False)

    if 'timeline' in settings:
        total_time = settings['timeline']['timeline'][-1][0]

    # run simulation
    experiment.update(total_time)

    # return data from emitter
    if return_raw_data:
        return experiment.emitter.get_data()
    else:
        return experiment.emitter.get_timeseries()


# plotting functions
def plot_compartment_topology(compartment, settings, out_dir='out', filename='topology'):
    """
    Make a plot of the topology
     - compartment: a compartment
    """
    store_rgb = [x/255 for x in [239,131,148]]
    process_rgb = [x / 255 for x in [249, 204, 86]]
    node_size = 4500
    font_size = 8
    node_distance = 1.5
    buffer = 0.2
    label_pos = 0.75

    network = compartment.generate({})
    topology = network['topology']
    processes = network['processes']

    # get figure settings
    show_ports = settings.get('show_ports', True)

    # make graph from topology
    G = nx.Graph()
    process_nodes = []
    store_nodes = []
    edges = {}
    for process_id, connections in topology.items():
        process_nodes.append(process_id)
        G.add_node(process_id)

        for port, store_id in connections.items():
            if store_id not in store_nodes:
                store_nodes.append(store_id)
            if store_id not in list(G.nodes):
                G.add_node(store_id)

            edge = (process_id, store_id)
            edges[edge] = port
            G.add_edge(process_id, store_id)

    # are there overlapping names?
    overlap = [name for name in process_nodes if name in store_nodes]
    if overlap:
        print('{} shared by processes and stores'.format(overlap))

    # get positions
    pos = {}
    n_rows = max(len(process_nodes), len(store_nodes))
    plt.figure(1, figsize=(10, n_rows * node_distance))

    for idx, node_id in enumerate(process_nodes, 1):
        pos[node_id] = np.array([-1, -idx])
    for idx, node_id in enumerate(store_nodes, 1):
        pos[node_id] = np.array([1, -idx])

    # plot
    nx.draw_networkx_nodes(G, pos,
                           nodelist=process_nodes,
                           node_color=process_rgb,
                           node_size=node_size,
                           node_shape='s')
    nx.draw_networkx_nodes(G, pos,
                           nodelist=store_nodes,
                           node_color=store_rgb,
                           node_size=node_size,
                           node_shape='o')
    # edges
    colors = list(range(1,len(edges)+1))
    nx.draw_networkx_edges(G, pos,
                           edge_color=colors,
                           width=1.5)
    # labels
    nx.draw_networkx_labels(G, pos,
                            font_size=font_size)
    if show_ports:
        nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=edges,
                                 font_size=font_size,
                                 label_pos=label_pos)

    # add buffer
    xmin, xmax, ymin, ymax = plt.axis()
    plt.xlim(xmin - buffer, xmax + buffer)
    plt.ylim(ymin - buffer, ymax + buffer)

    # save figure
    fig_path = os.path.join(out_dir, filename)
    plt.axis('off')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def set_axes(ax, show_xaxis=False):
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-5,5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(right=False, top=False)

    # move offset axis text (typically scientific notation)
    t = ax.yaxis.get_offset_text()
    t.set_x(-0.4)
    if not show_xaxis:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)

def plot_simulation_output(timeseries_raw, settings={}, out_dir='out', filename='simulation'):
    '''
    Plot simulation output, with rows organized into separate columns.

    Arguments::
        timeseries (dict): This can be obtained from simulation output with convert_to_timeseries()
        settings (dict): Accepts the following keys:

            * **max_rows** (:py:class:`int`): ports with more states
              than this number of states get wrapped into a new column
            * **remove_zeros** (:py:class:`bool`): if True, timeseries
              with all zeros get removed
            * **remove_flat** (:py:class:`bool`): if True, timeseries
              with all the same value get removed
            * **remove_first_timestep** (:py:class:`bool`): if True,
              skips the first timestep
            * **skip_ports** (:py:class:`list`): entire ports that won't
              be plotted
            * **show_state** (:py:class:`list`): with
              ``[('port_id', 'state_id')]`` for all states that will be
              highlighted, even if they are otherwise to be removed
    '''

    plot_fontsize = 8
    plt.rc('font', size=plot_fontsize)
    plt.rc('axes', titlesize=plot_fontsize)

    skip_keys = ['time']

    # get settings
    max_rows = settings.get('max_rows', 25)
    remove_zeros = settings.get('remove_zeros', True)
    remove_flat = settings.get('remove_flat', False)
    skip_ports = settings.get('skip_ports', [])
    remove_first_timestep = settings.get('remove_first_timestep', False)

    # make a flat 'path' timeseries, with keys being path
    top_level = list(timeseries_raw.keys())
    timeseries = path_timeseries_from_embedded_timeseries(timeseries_raw)
    time_vec = timeseries.pop('time')
    if remove_first_timestep:
        time_vec = time_vec[1:]

    # remove select states from timeseries
    removed_states = set()
    for path, series in timeseries.items():
        if path[0] in skip_ports:
            removed_states.add(path)
        elif remove_flat:
            if series.count(series[0]) == len(series):
                removed_states.add(path)
        elif remove_zeros:
            if all(v == 0 for v in series):
                removed_states.add(path)
    for path in removed_states:
        del timeseries[path]

    ## get figure columns
    # get length of each top-level port
    port_lengths = {}
    for path in timeseries.keys():
        if path[0] in top_level:
            if path[0] not in port_lengths:
                port_lengths[path[0]] = 0
            port_lengths[path[0]] += 1
    n_data = [length for port, length in port_lengths.items() if length > 0]
    columns = []
    for n_states in n_data:
        new_cols = n_states / max_rows
        if new_cols > 1:
            for col in range(int(new_cols)):
                columns.append(max_rows)
            mod_states = n_states % max_rows
            if mod_states > 0:
                columns.append(mod_states)
        else:
            columns.append(n_states)

    # make figure and plot
    n_cols = len(columns)
    n_rows = max(columns)
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 1))
    grid = plt.GridSpec(n_rows, n_cols)
    row_idx = 0
    col_idx = 0
    for port in port_lengths.keys():
        # get this port's states
        port_timeseries = {path[1:]: ts for path, ts in timeseries.items() if path[0] is port}
        for state_id, series in sorted(port_timeseries.items()):
            if remove_first_timestep:
                series = series[1:]
            # not enough data points -- this state likely did not exist throughout the entire simulation
            if len(series) != len(time_vec):
                continue

            ax = fig.add_subplot(grid[row_idx, col_idx])  # grid is (row, column)

            if not all(isinstance(state, (int, float, np.int64, np.int32)) for state in series):
                # check if series is a list of ints or floats
                ax.title.set_text(str(port) + ': ' + str(state_id) + ' (non numeric)')
            else:
                # plot line at zero if series crosses the zero line
                if any(x == 0.0 for x in series) or (any(x < 0.0 for x in series) and any(x > 0.0 for x in series)):
                    zero_line = [0 for t in time_vec]
                    ax.plot(time_vec, zero_line, 'k--')

                # plot the series
                ax.plot(time_vec, series)
                ax.title.set_text(str(port) + ': ' + str(state_id))

            if row_idx == columns[col_idx]-1:
                # if last row of column
                set_axes(ax, True)
                ax.set_xlabel('time (s)')
                row_idx = 0
                col_idx += 1
            else:
                set_axes(ax)
                row_idx += 1
            ax.set_xlim([time_vec[0], time_vec[-1]])

    # save figure
    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.8, hspace=1.0)
    plt.savefig(fig_path, bbox_inches='tight')



def order_list_of_paths(path_list):
    # make the lists equal in length:
    length = max(map(len, path_list))
    lol = np.array([list(path) + [None] * (length - len(path)) for path in path_list])

    # sort by first two columns. TODO -- sort by all available columns
    ind = np.lexsort((lol[:, 1], lol[:, 0]))
    sorted_path_list = sorted(zip(ind, path_list))
    forward_order = [idx_path[1] for idx_path in sorted_path_list]
    forward_order.reverse()
    return forward_order

def plot_agents_multigen(data, settings={}, out_dir='out', filename='agents'):
    '''
    Plot multi-agent simulation output, with all agents data combined for every
    corresponding path in their stores.
    Arguments::
        data (dict): This is raw_data obtained from simulation output
        settings (dict): Accepts the following keys:

            * **max_rows** (:py:class:`int`): ports with more states
              than this number of states get wrapped into a new column
            * **remove_zeros** (:py:class:`bool`): if True, timeseries
              with all zeros get removed
            * **remove_flat** (:py:class:`bool`): if True, timeseries
              with all the same value get removed
            * **skip_paths** (:py:class:`list`): entire path, including subpaths
              that won't be plotted
            * **include_paths** (:py:class:`list`): list of full paths
              to include. Overridden by skip_paths.
            * **titles_map** (:py:class:`dict`): Map from path tuples to
              strings to use as the figure titles for each path's plot.
              If not provided, the path is shown as the title.
            * **ylabels_map** (:py:class:`dict`): Map from path tuples to
              strings to use as the y-axis labels for each path's plot.
              If not specified, no y-axis label is used.

    TODO -- add legend with agent color
    '''

    agents_key = settings.get('agents_key', 'agents')
    max_rows = settings.get('max_rows', 25)
    remove_zeros = settings.get('remove_zeros', False)
    remove_flat = settings.get('remove_flat', False)
    skip_paths = settings.get('skip_paths', [])
    include_paths = settings.get('include_paths', None)
    title_size = settings.get('title_size', 16)
    tick_label_size = settings.get('tick_label_size', 12)
    titles_map = settings.get('titles_map', dict())
    ylabels_map = settings.get('ylabels_map', dict())
    time_vec = list(data.keys())
    timeseries = path_timeseries_from_data(data)

    # get the agents' port_schema in a set of paths.
    # this assumes that the initial agent's schema and behavior
    # is representative of later agents
    initial_agents = data[time_vec[0]][agents_key]
    # make the set of paths
    if include_paths is None:
        port_schema_paths = set()
        for agent_id, agent_data in initial_agents.items():
            path_list = get_path_list_from_dict(agent_data)
            port_schema_paths.update(path_list)
    else:
        port_schema_paths = set(include_paths)
    # make set of paths to remove
    remove_paths = set()
    for path, series in timeseries.items():
        if path[0] == agents_key and path[1] in list(initial_agents.keys()):
            agent_path = path[2:]
            if remove_flat:
                if series.count(series[0]) == len(series):
                    remove_paths.add(agent_path)
            elif remove_zeros:
                if all(v == 0 for v in series):
                    remove_paths.add(agent_path)
    # get paths and subpaths from skip_paths to remove
    for path in port_schema_paths:
        for remove in skip_paths:
            if set(path) >= set(remove):
                remove_paths.add(path)
    # remove the paths
    port_schema_paths = [path for path in port_schema_paths if path not in remove_paths]
    top_ports = set([path[0] for path in port_schema_paths])

    # get port columns, assign subplot locations
    port_rows = {port_id: [] for port_id in top_ports}
    for path in port_schema_paths:
        top_port = path[0]
        port_rows[top_port].append(path)

    highest_row = 0
    row_idx = 0
    col_idx = 0
    ordered_paths = {port_id: {} for port_id in top_ports}
    for port_id, path_list in port_rows.items():
        if not path_list:
            continue
        # order target names and assign subplot location
        ordered_targets = order_list_of_paths(path_list)
        for target in ordered_targets:
            ordered_paths[port_id][target] = [row_idx, col_idx]

            # next column/row
            if row_idx >= max_rows - 1:
                row_idx = 0
                col_idx += 1
            else:
                row_idx += 1
            if row_idx > highest_row:
                highest_row = row_idx
        # new column for next port
        row_idx = 0
        col_idx += 1

    # initialize figure
    n_rows = highest_row + 1
    n_cols = col_idx + 1
    fig = plt.figure(figsize=(4 * n_cols, 2 * n_rows))
    grid = plt.GridSpec(ncols=n_cols, nrows=n_rows, wspace=0.4, hspace=1.5)

    # make the subplot axes
    port_axes = {}
    for port_id, paths in ordered_paths.items():
        for path_idx, (path, location) in enumerate(paths.items()):
            row_idx = location[0]
            col_idx = location[1]

            # make the subplot axis
            ax = fig.add_subplot(grid[row_idx, col_idx])
            for tick_type in ('major', 'minor'):
                ax.tick_params(
                    axis='both',
                    which=tick_type,
                    labelsize=tick_label_size,
                )
            ax.title.set_text(titles_map.get(path, path))
            ax.title.set_fontsize(title_size)
            if path in ylabels_map:
                ax.set_ylabel(ylabels_map[path], fontsize=title_size)
            ax.set_xlim([time_vec[0], time_vec[-1]])
            ax.xaxis.get_offset_text().set_fontsize(tick_label_size)
            ax.yaxis.get_offset_text().set_fontsize(tick_label_size)

            # if last state in this port, add time ticks
            if (row_idx >= highest_row
                or path_idx >= len(ordered_paths[port_id]) - 1
            ):
                set_axes(ax, True)
                ax.set_xlabel('time (s)', fontsize=title_size)

            else:
                set_axes(ax)
            ax.set_xlim([time_vec[0], time_vec[-1]])
            # save axis
            port_axes[path] = ax

    # plot the agents
    plotted_agents = []
    for time_idx, (time, time_data) in enumerate(data.items()):
        agents = time_data[agents_key]
        for agent_id, agent_data in agents.items():
            if agent_id not in plotted_agents:
                plotted_agents.append(agent_id)
                for port_schema_path in port_schema_paths:
                    agent_port_schema_path = (agents_key, agent_id) + port_schema_path
                    if agent_port_schema_path not in timeseries:
                        continue

                    series = timeseries[agent_port_schema_path]
                    if not isinstance(series[0], (float, int)):
                        continue
                    n_times = len(series)
                    plot_times = time_vec[time_idx:time_idx+n_times]

                    ax = port_axes[port_schema_path]
                    ax.plot(plot_times, series)

    # save figure
    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig(fig_path, bbox_inches='tight')


# timeseries functions
def agent_timeseries_from_data(data, agents_key='cells'):
    timeseries = {}
    for time, all_states in data.items():
        agent_data = all_states[agents_key]
        for agent_id, ports in agent_data.items():
            if agent_id not in timeseries:
                timeseries[agent_id] = {}
            for port_id, states in ports.items():
                if port_id not in timeseries[agent_id]:
                    timeseries[agent_id][port_id] = {}
                for state_id, state in states.items():
                    if state_id not in timeseries[agent_id][port_id]:
                        timeseries[agent_id][port_id][state_id] = []
                    timeseries[agent_id][port_id][state_id].append(state)
    return timeseries

def save_timeseries(timeseries, out_dir='out'):
    flattened = flatten_timeseries(timeseries)
    save_flat_timeseries(flattened, out_dir)

def save_flat_timeseries(timeseries, out_dir='out'):
    '''Save a timeseries as a CSV in out_dir'''
    rows = np.transpose(list(timeseries.values())).tolist()
    with open(os.path.join(out_dir, 'simulation_data.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(timeseries.keys())
        writer.writerows(rows)

def load_timeseries(path_to_csv):
    '''Load a timeseries saved as a CSV using save_timeseries.

    The timeseries is returned in flattened form.
    '''
    with io.open(path_to_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        timeseries = {}
        for row in reader:
            for header, elem in row.items():
                if elem == '':
                    elem = None
                if elem is not None:
                    elem = float(elem)
                timeseries.setdefault(header, []).append(elem)
    return timeseries

def timeseries_to_ndarrays(timeseries, keys=None):
    '''After filtering by keys, convert timeseries to dict of ndarrays

    Returns:
        dict: Mapping from timeseries variables to an ndarray of the
            variable values.
    '''
    if keys is None:
        keys = timeseries.keys()
    return {
        key: np.array(timeseries[key], dtype=np.float) for key in keys}

def _prepare_timeseries_for_comparison(
    timeseries1, timeseries2, keys=None,
    required_frac_checked=0.9,
):
    '''Prepare two timeseries for comparison

    Arguments:
        timeseries1: One timeseries. Must be flattened and include times
            under the 'time' key.
        timeseries2: The other timeseries. Same requirements as
            timeseries1.
        keys: Keys of the timeseries whose values will be checked for
            correlation. If not specified, all keys present in both
            timeseries are used.
        required_frac_checked: The required fraction of timepoints in a
            timeseries that must be checked. If this requirement is not
            satisfied, which might occur if the two timeseries share few
            timepoints, the test wll fail.

    Returns:
        A tuple of an ndarray for each of the two timeseries and a list of
        the keys for the rows of the arrays. Each ndarray has a row for
        each key, in the order of keys. The ndarrays have only the
        columns corresponding to the timepoints common to both
        timeseries.

    Raises:
        AssertionError: If a correlation is strictly below the
            threshold or if too few timepoints are common to both
            timeseries.
    '''
    if 'time' not in timeseries1 or 'time' not in timeseries2:
        raise AssertionError('Both timeseries must have key "time"')
    if keys is None:
        keys = set(timeseries1.keys()) & set(timeseries2.keys())
    else:
        if 'time' not in keys:
            keys.append('time')
    keys = list(keys)
    time_index = keys.index('time')
    shared_times = set(timeseries1['time']) & set(timeseries2['time'])
    frac_timepoints_checked = (
        len(shared_times)
        / min(len(timeseries1['time']), len(timeseries2['time']))
    )
    if frac_timepoints_checked < required_frac_checked:
        raise AssertionError(
            'The timeseries share too few timepoints: '
            '{} < {}'.format(
                frac_timepoints_checked, required_frac_checked)
        )
    masked = []
    for ts in (timeseries1, timeseries2):
        arrays_dict = timeseries_to_ndarrays(ts, keys)
        arrays_dict_shared_times = {}
        for key, array in arrays_dict.items():
            # Filters out times after data ends
            times_for_array = arrays_dict['time'][:len(array)]
            arrays_dict_shared_times[key] = array[
                np.isin(times_for_array, list(shared_times))]
        masked.append(arrays_dict_shared_times)
    return (
        masked[0],
        masked[1],
        keys,
    )

def assert_timeseries_correlated(
    timeseries1, timeseries2, keys=None,
    default_threshold=(1 - 1e-10), thresholds={},
    required_frac_checked=0.9,
):
    '''Check that two timeseries are correlated.

    Uses a Pearson correlation coefficient. Only the data from
    timepoints common to both timeseries are compared.

    Arguments:
        timeseries1: One timeseries. Must be flattened and include times
            under the 'time' key.
        timeseries2: The other timeseries. Same requirements as
            timeseries1.
        keys: Keys of the timeseries whose values will be checked for
            correlation. If not specified, all keys present in both
            timeseries are used.
        default_threshold: The threshold correlation coefficient to use
            when a threshold is not specified in thresholds.
        thresholds: Dictionary of key-value pairs where the key is a key
            in both timeseries and the value is the threshold
            correlation coefficient to use when checking that key
        required_frac_checked: The required fraction of timepoints in a
            timeseries that must be checked. If this requirement is not
            satisfied, which might occur if the two timeseries share few
            timepoints, the test wll fail. This is also the fraction of
            timepoints for each variable that must be non-nan in both
            timeseries. Note that the denominator of this fraction is
            the number of shared timepoints that are non-nan in either
            of the timeseries.

    Raises:
        AssertionError: If a correlation is strictly below the
            threshold or if too few timepoints are common to both
            timeseries.
    '''
    arrays1, arrays2, keys = _prepare_timeseries_for_comparison(
        timeseries1, timeseries2, keys, required_frac_checked)
    for key in keys:
        both_nan = np.isnan(arrays1[key]) & np.isnan(arrays2[key])
        valid_indices = ~(
            np.isnan(arrays1[key]) | np.isnan(arrays2[key]))
        frac_checked = valid_indices.sum() / (~both_nan).sum()
        if frac_checked < required_frac_checked:
            raise AssertionError(
                'Timeseries share too few non-nan values for variable '
                '{}: {} < {}'.format(
                    key, frac_checked, required_frac_checked
                )
            )
        corrcoef = np.corrcoef(
            arrays1[key][valid_indices],
            arrays2[key][valid_indices],
        )[0][1]
        threshold = thresholds.get(key, default_threshold)
        if corrcoef < threshold:
            raise AssertionError(
                'The correlation coefficient for '
                '{} is too small: {} < {}'.format(
                    key, corrcoef, threshold)
            )

def assert_timeseries_close(
    timeseries1, timeseries2, keys=None,
    default_tolerance=(1 - 1e-10), tolerances={},
    required_frac_checked=0.9,
):
    '''Check that two timeseries are similar.

    Ensures that each pair of data points between the two timeseries are
    within a tolerance of each other, after filtering out timepoints not
    common to both timeseries.

    Arguments:
        timeseries1: One timeseries. Must be flattened and include times
            under the 'time' key.
        timeseries2: The other timeseries. Same requirements as
            timeseries1.
        keys: Keys of the timeseries whose values will be checked for
            correlation. If not specified, all keys present in both
            timeseries are used.
        default_tolerance: The tolerance to use when not specified in
            tolerances.
        tolerances: Dictionary of key-value pairs where the key is a key
            in both timeseries and the value is the tolerance to use
            when checking that key.
        required_frac_checked: The required fraction of timepoints in a
            timeseries that must be checked. If this requirement is not
            satisfied, which might occur if the two timeseries share few
            timepoints, the test wll fail.

    Raises:
        AssertionError: If a pair of data points have a difference
            strictly above the tolerance threshold or if too few
            timepoints are common to both timeseries.
    '''
    arrays1, arrays2, keys = _prepare_timeseries_for_comparison(
        timeseries1, timeseries2, keys, required_frac_checked)
    for key in keys:
        tolerance = tolerances.get(key, default_tolerance)
        close_mask = np.isclose(arrays1[key], arrays2[key],
            atol=tolerance, equal_nan=True)
        if not np.all(close_mask):
            print('Timeseries 1:', arrays1[key][~close_mask])
            print('Timeseries 2:', arrays2[key][~close_mask])
            raise AssertionError(
                'The data for {} differed by more than {}'.format(
                    key, tolerance)
            )


# TESTS
class ToyLinearGrowthDeathProcess(Process):

    name = 'toy_linear_growth_death'

    GROWTH_RATE = 1.0
    THRESHOLD = 6.0

    def __init__(self, initial_parameters={}):
        self.targets = initial_parameters.get('targets')
        super(ToyLinearGrowthDeathProcess, self).__init__(initial_parameters)

    def ports_schema(self):
        return {
            'global': {
                'mass': {
                    '_default': 1.0,
                    '_emit': True}},
            'targets': {
                target: {
                    '_default': None}
                for target in self.targets}}

    def next_update(self, timestep, states):
        mass = states['global']['mass']
        mass_grown = (
            ToyLinearGrowthDeathProcess.GROWTH_RATE * timestep)
        update = {
            'global': {'mass': mass_grown},
        }
        if mass > ToyLinearGrowthDeathProcess.THRESHOLD:
            update['global'] = {
                '_delete': [(target,) for target in self.targets]}

        return update


class TestSimulateProcess:

    def test_process_deletion(self):
        '''Check that processes are successfully deleted'''
        process = ToyLinearGrowthDeathProcess({'targets': ['process']})
        settings = {
            'emit_step': 1,
            'topology': {
                'global': ('global',),
                'targets': tuple()}}

        timeseries = simulate_process(process, settings)
        expected_masses = [
            # Mass stops increasing the iteration after mass > 5 because
            # cell dies
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0]
        masses = timeseries['global']['mass']
        assert masses == expected_masses


# toy processes
class ToyMetabolism(Process):
    name = 'toy_metabolism'

    def __init__(self, initial_parameters={}):
        parameters = {'mass_conversion_rate': 1}
        parameters.update(initial_parameters)
        super(ToyMetabolism, self).__init__(parameters)

    def ports_schema(self):
        ports = {
            'pool': ['GLC', 'MASS']}
        return {
            port_id: {
                key: {
                    '_default': 0.0,
                    '_emit': True}
                for key in keys}
            for port_id, keys in ports.items()}

    def next_update(self, timestep, states):
        update = {}
        glucose_required = timestep / self.parameters['mass_conversion_rate']
        if states['pool']['GLC'] >= glucose_required:
            update = {
                'pool': {
                    'GLC': -2,
                    'MASS': 1}}

        return update

class ToyTransport(Process):
    name = 'toy_transport'

    def __init__(self, initial_parameters={}):
        parameters = {'intake_rate': 2}
        parameters.update(initial_parameters)
        super(ToyTransport, self).__init__(parameters)

    def ports_schema(self):
        ports = {
            'external': ['GLC'],
            'internal': ['GLC']}
        return {
            port_id: {
                key: {
                    '_default': 0.0,
                    '_emit': True}
                for key in keys}
            for port_id, keys in ports.items()}

    def next_update(self, timestep, states):
        update = {}
        intake = timestep * self.parameters['intake_rate']
        if states['external']['GLC'] >= intake:
            update = {
                'external': {'GLC': -2, 'MASS': 1},
                'internal': {'GLC': 2}}

        return update

class ToyDeriveVolume(Deriver):
    name = 'toy_derive_volume'

    def __init__(self, initial_parameters={}):
        parameters = {}
        super(ToyDeriveVolume, self).__init__(parameters)

    def ports_schema(self):
        ports = {
            'compartment': ['MASS', 'DENSITY', 'VOLUME']}
        return {
            port_id: {
                key: {
                    '_updater': 'set' if key == 'VOLUME' else 'accumulate',
                    '_default': 0.0,
                    '_emit': True}
                for key in keys}
            for port_id, keys in ports.items()}

    def next_update(self, timestep, states):
        volume = states['compartment']['MASS'] / states['compartment']['DENSITY']
        update = {
            'compartment': {'VOLUME': volume}}

        return update

class ToyDeath(Process):
    name = 'toy_death'

    def __init__(self, initial_parameters={}):
        self.targets = initial_parameters.get('targets', [])
        super(ToyDeath, self).__init__({})

    def ports_schema(self):
        return {
            'compartment': {
                'VOLUME': {
                    '_default': 0.0,
                    '_emit': True}},
            'global': {
                target: {
                    '_default': None}
                for target in self.targets}}

    def next_update(self, timestep, states):
        volume = states['compartment']['VOLUME']
        update = {}

        if volume > 1.0:
            # kill the cell
            update = {
                'global': {
                    '_delete': [
                        (target,)
                        for target in self.targets]}}

        return update

class ToyCompartment(Generator):
    '''
    a toy compartment for testing

    '''
    def __init__(self, config):
        super(ToyCompartment, self).__init__(config)

    def generate_processes(self, config):
        return {
            'metabolism': ToyMetabolism(
                {'mass_conversion_rate': 0.5}), # example of overriding default parameters
            'transport': ToyTransport(),
            'death': ToyDeath({'targets': [
                'metabolism',
                'transport']}),
            'external_volume': ToyDeriveVolume(),
            'internal_volume': ToyDeriveVolume()
        }

    def generate_topology(self, config):
        return{
            'metabolism': {
                'pool': ('cytoplasm',)},
            'transport': {
                'external': ('periplasm',),
                'internal': ('cytoplasm',)},
            'death': {
                'global': tuple(),
                'compartment': ('cytoplasm',)},
            'external_volume': {
                'compartment': ('periplasm',)},
            'internal_volume': {
                'compartment': ('cytoplasm',)}}


def test_compartment():
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


if __name__ == '__main__':
    TestSimulateProcess().test_process_deletion()
    timeseries = test_compartment()
    print(timeseries)
