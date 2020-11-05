import os

import numpy as np
import matplotlib.pyplot as plt

from vivarium.core.emitter import path_timeseries_from_embedded_timeseries


def set_axes(ax, show_xaxis=False):
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-5, 5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(right=False, top=False)

    # move offset axis text (typically scientific notation)
    t = ax.yaxis.get_offset_text()
    t.set_x(-0.4)
    if not show_xaxis:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)


def plot_simulation_output(
        timeseries_raw,
        settings={},
        out_dir=None,
        filename=None,
):
    '''
    Plot simulation output, with rows organized into separate columns.

    Arguments::
        timeseries (dict): This can be obtained from simulation output with convert_to_timeseries()
        settings (dict): Accepts the following keys:

            * **column_width** (:py:class:`int`): the width (inches) of
              each column in the figure
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

    # get settings
    column_width = settings.get('column_width', 3)
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
    fig = plt.figure(figsize=(n_cols * column_width, n_rows * column_width/3))
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

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            filename = 'simulation'
        # save figure
        fig_path = os.path.join(out_dir, filename)
        plt.subplots_adjust(wspace=column_width/3, hspace=column_width/3)
        plt.savefig(fig_path, bbox_inches='tight')
    else:
        return fig
