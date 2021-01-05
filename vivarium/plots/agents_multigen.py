import os

import numpy as np
import matplotlib.pyplot as plt

from vivarium.core.emitter import path_timeseries_from_data
from vivarium.library.dict_utils import get_path_list_from_dict
from vivarium.plots.simulation_output import set_axes


def order_list_of_paths(path_list):
    # make the lists equal in length:
    length = max(map(len, path_list))
    lol = np.array([list(path) + [None] * (length - len(path)) for path in path_list])

    if lol.shape[0] > 1:
        # sort by first two columns. TODO -- sort by all available columns
        ind = np.lexsort((lol[:, 1], lol[:, 0]))
        sorted_path_list = sorted(zip(ind, path_list))
        forward_order = [idx_path[1] for idx_path in sorted_path_list]
        forward_order.reverse()
        return forward_order
    else:
        return path_list


def plot_agents_multigen(
        data,
        settings={},
        out_dir=None,
        filename=None,
):
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
        out_dir (str): TODO
        filename (str): TODO

    TODO -- add legend with agent color
    '''

    agents_key = settings.get('agents_key', 'agents')
    max_rows = settings.get('max_rows', 25)
    column_width = settings.get('column_width', 4)
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
        if len(path_list) > 1:
            sorted_path = sorted(path_list, key=lambda x: x[1])
            port_rows[port_id] = sorted_path
        else:
            port_rows[port_id] = path_list

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
    fig = plt.figure(figsize=(column_width * n_cols, column_width/2 * n_rows))
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
        if agents_key not in time_data:
            print('{} key missing at time {}'.format(agents_key, time))
        else:
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

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            filename = 'agents'
        fig_path = os.path.join(out_dir, filename)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.savefig(fig_path, bbox_inches='tight')
    else:
        return fig
