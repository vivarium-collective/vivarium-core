from __future__ import print_function, division, absolute_import

import csv
import os


def process_path_timeseries_for_csv(path_ts):
    # type: (dict) -> dict
    '''Prepare path timeseries data for writing to CSV

    Processing Steps:

    1. Convert any dictionary keys that are tuples to strings by joining
       the tuple elements with ``-``.
    2. Remove from the timeseries any data where each timepoint is not
       numeric. For example, we remove data where each timepoint is a
       matrix or a string.

    .. note:: We assume that across timepoints for the same variable,
        the data types are the same.

    Returns:
        dict: A timeseries that can be saved to a CSV with
        :py:func:`save_flat_timeseries`.
    '''
    # Convert tuple keys to strings
    str_keys = dict()
    for key, value in path_ts.items():
        try:
            iter(key)
            if not isinstance(key, str):
                key = ",".join(key)
        except TypeError:
            pass
        str_keys[key] = value

    remove_keys = [
        # Remove matrices
        key for key, val in str_keys.items()
        if isinstance(val[0], list)
    ]
    # Remove keys with non-numeric data
    for key, val in str_keys.items():
        try:
            float(val[0])
        except (ValueError, TypeError):
            remove_keys.append(key)
    for key in set(remove_keys):
        del str_keys[key]
    return str_keys


def save_flat_timeseries(timeseries, out_dir, filename):
    # type: (dict, str, str) -> None
    '''Save a flattened timeseries to a CSV file

    The CSV file will have one column for each key in the timeseries.
    The heading will be the key, and the rows will contain the
    timeseries data, one row per timepoint, in increasing order of time.
    '''
    n_rows = max([len(val) for val in timeseries.values()])
    rows = [{} for _ in range(n_rows)]
    for key, val in timeseries.items():
        for i, elem in enumerate(val):
            rows[i][key] = elem
    with open(os.path.join(out_dir, filename), 'w') as f:
        writer = csv.DictWriter(f, timeseries.keys(), delimiter=',')
        writer.writeheader()
        writer.writerows(rows)
