"""
==================================
Plotting Functions for Experiments
==================================
"""


import os

import numpy as np
import matplotlib.pyplot as plt

from vivarium.core.emitter import path_timeseries_from_embedded_timeseries, path_timeseries_from_data
from vivarium.library.dict_utils import get_path_list_from_dict


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
    os.makedirs(out_dir, exist_ok=True)

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
                if isinstance(state_id, tuple):
                    # new line for each store
                    state_id = '\n'.join(state_id)
                ax.title.set_text(str(port) + '\n' + str(state_id))

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


def plot_agents_multigen(data, settings={}, out_dir='out', filename='agents'):
    '''Plot values over time for multiple agents and generations

    Plot multi-agent simulation output, with all agents data combined for every
    corresponding path in their stores.

    Arguments:
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

    # get list of states for each port
    port_rows = {port_id: [] for port_id in top_ports}
    for path in port_schema_paths:
        top_port = path[0]
        port_rows[top_port].append(path)

    # sort each port by second element
    for port_id, path_list in port_rows.items():
        sorted_path = sorted(path_list, key=lambda x: x[1])
        port_rows[port_id] = sorted_path

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
            state_title = titles_map.get(path, path)
            if isinstance(state_title, tuple):
                # new line for each store
                state_title = ' \n'.join(state_title)
            ax.title.set_text(state_title)
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
