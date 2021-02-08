import csv
import io
import os
from typing import Dict, List, Optional, Any

import numpy as np

from vivarium.library.dict_utils import flatten_timeseries


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
    rows: List[Dict] = [{} for _ in range(n_rows)]
    for key, val in timeseries.items():
        for i, elem in enumerate(val):
            rows[i][key] = elem
    with open(os.path.join(out_dir, filename), 'w') as f:
        writer = csv.DictWriter(f, timeseries.keys(), delimiter=',')
        writer.writeheader()
        writer.writerows(rows)


def assert_timeseries_close(
    timeseries1, timeseries2, keys=None,
    default_tolerance=(1 - 1e-10), tolerances: Optional[Dict[str, Any]] = None,
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
    tolerances = tolerances or {}
    arrays1, arrays2, keys = _prepare_timeseries_for_comparison(
        timeseries1, timeseries2, keys, required_frac_checked)
    for key in keys:
        tolerance = tolerances.get(key, default_tolerance)
        close_mask = np.isclose(
            arrays1[key], arrays2[key], atol=tolerance, equal_nan=True)
        if not np.all(close_mask):
            print('Timeseries 1:', arrays1[key][~close_mask])
            print('Timeseries 2:', arrays2[key][~close_mask])
            raise AssertionError(
                'The data for {} differed by more than {}'.format(
                    key, tolerance)
            )


def assert_timeseries_correlated(
    timeseries1, timeseries2, keys=None,
    default_threshold=(1 - 1e-10), thresholds: Optional[Dict[str, Any]] = None,
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
    thresholds = thresholds or {}
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


def timeseries_to_ndarrays(timeseries, keys=None):
    '''After filtering by keys, convert timeseries to dict of ndarrays

    Returns:
        dict: Mapping from timeseries variables to an ndarray of the
            variable values.
    '''
    if keys is None:
        keys = timeseries.keys()
    return {
        key: np.array(timeseries[key], dtype=float) for key in keys}


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


def save_timeseries(timeseries, out_dir='out', filename='timeseries.csv'):
    flattened = flatten_timeseries(timeseries)
    save_flat_timeseries(flattened, out_dir, filename)


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
