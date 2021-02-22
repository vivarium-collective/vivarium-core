
import collections.abc
import copy
from functools import reduce
import operator
import traceback
from typing import Optional

from vivarium.library.units import Quantity


MULTI_UPDATE_KEY = '_multi_update'

tuple_separator = '___'


def merge_dicts(dicts):
    merge = {}
    for d in dicts:
        merge.update(d)
    return merge


def deep_merge_check(dct, merge_dct):
    """Recursive dict merge with checks

    Throws exceptions for conflicting values.
    This mutates dct - the contents of merge_dct are added to dct (which is also returned).
    If you want to keep dct you could call it like deep_merge(dict(dct), merge_dct)
    """

    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            try:
                deep_merge_check(dct[k], merge_dct[k])
            except ValueError:
                raise ValueError('dict merge mismatch: key "{}" has values {} AND {}'.format(k, dct[k], merge_dct[k]))
        elif k in dct and (dct[k] is not merge_dct[k]):
            raise ValueError('dict merge mismatch: key "{}" has values {} AND {}'.format(k, dct[k], merge_dct[k]))
        else:
            dct[k] = merge_dct[k]
    return dct


def deep_merge_combine_lists(dct, merge_dct):
    """ Recursive dict merge with lists

    Values that are lists are combined into one list without repeating values.
    This mutates dct - the contents of merge_dct are added to dct (which is also returned).
    If you want to keep dct you could call it like deep_merge(dict(dct), merge_dct)
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            deep_merge(dct[k], merge_dct[k])
        elif k in dct and isinstance(dct[k], list) and isinstance(v, list):
            for i in v:
                if i not in dct[k]:
                    dct[k].append(i)
        else:
            dct[k] = merge_dct[k]
    return dct


def deep_merge_multi_update(dct, merge_dct):
    """ Recursive dict merge combines multiple values

    If a value already exists for a key, it is added in a list
    """
    if dct is None:
        dct = {}
    if merge_dct is None:
        merge_dct = {}
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            deep_merge_multi_update(dct[k], merge_dct[k])
        elif k in dct:
            # put values together in a list under '_multi_update' key
            if isinstance(dct[k], dict) and MULTI_UPDATE_KEY in dct[k]:
                dct[k]['_multi_update'].append(merge_dct[k])
            else:
                dct[k] = {
                    '_multi_update': [
                        dct[k], merge_dct[k]]}
        else:
            dct[k] = merge_dct[k]
    return dct


def deep_merge(dct, merge_dct):
    """ Recursive dict merge

    This mutates dct - the contents of merge_dct are added to dct (which is also returned).
    If you want to keep dct you could call it like deep_merge(dict(dct), merge_dct)
    """
    if dct is None:
        dct = {}
    if merge_dct is None:
        merge_dct = {}
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            deep_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def flatten_port_dicts(dicts):
    """
    Input:
        dicts (dict): embedded state dictionaries with the {'port_id': {'state_id': state_value}}

    Return:
        dict: flattened dictionary with {'state_id_port_id': value}
    """
    merge = {}
    for port, states_dict in dicts.items():
        for state, value in states_dict.items():
            merge.update({state + '_' + port: value})
    return merge


def tuplify_port_dicts(dicts):
    """
    Input:
        dicts (dict): embedded state dictionaries with the {'port_id': {'state_id': state_value}}

    Return:
        dict: tuplified dictionary with {(port_id','state_id'): value}
    """
    merge = {}
    for port, states_dict in dicts.items():
        if states_dict:
            for state, value in states_dict.items():
                merge.update({(port, state): value})
    return merge


def flatten_timeseries(timeseries):
    """Flatten a timeseries in the style of flatten_port_dicts"""
    flat = {}
    for port, store_dict in timeseries.items():
        if port == 'time':
            flat[port] = timeseries[port]
            continue
        for variable_name, values in store_dict.items():
            key = "{}_{}".format(port, variable_name)
            flat[key] = values
    return flat


def tuple_to_str_keys(dictionary):
    """
    take a dict with tuple keys, and convert them to strings with tuple_separator as a delimiter
    """
    new_dict = copy.deepcopy(dictionary)
    make_str_dict(new_dict)
    return new_dict


def make_str_dict(dictionary):
    # get down to the leaves first
    for k, v in dictionary.items():
        if isinstance(v, dict):
            make_str_dict(v)

        # convert tuples in lists
        if isinstance(v, list):
            for idx, var in enumerate(v):
                if isinstance(var, tuple):
                    v[idx] = tuple_separator.join(var)
                if isinstance(var, dict):
                    make_str_dict(var)

    # which keys are tuples?
    tuple_ks = [k for k in dictionary.keys() if isinstance(k, tuple)]
    for tuple_k in tuple_ks:
        str_k = tuple_separator.join(tuple_k)
        dictionary[str_k] = dictionary[tuple_k]
        del dictionary[tuple_k]

    return dictionary


def str_to_tuple_keys(dictionary):
    """
    Take a dict with keys that have tuple_separator, and convert them to tuples
    """

    # get down to the leaves first
    for k, v in dictionary.items():
        if isinstance(v, dict):
            str_to_tuple_keys(v)

        # convert strings in lists
        if isinstance(v, list):
            for idx, var in enumerate(v):
                if isinstance(var, str) and tuple_separator in var:
                    v[idx] = tuple(var.split(tuple_separator))
                if isinstance(var, dict):
                    str_to_tuple_keys(var)

    # which keys are tuples?
    str_ks = [k for k in dictionary.keys() if isinstance(k, str) and tuple_separator in k]
    for str_k in str_ks:
        tuple_k = tuple(str_k.split(tuple_separator))
        dictionary[tuple_k] = dictionary[str_k]
        del dictionary[str_k]

    return dictionary


def keys_list(d: dict) -> list:
    """Return list(d.keys())."""
    return list(d.keys())


def value_in_embedded_dict(
        data: dict,
        timeseries: dict = None,
        time_index: Optional[float] = None) -> dict:
    """
    converts data from a single time step into an embedded dictionary with lists
    of values.
    If the value has a unit, saves under a key with (key, unit_string).
    """
    # TODO(jerry): ^^^ Explain this further. Note that this function modifies
    #  timeseries.
    # TODO(jerry): Refine the type declarations.
    # TODO(jerry): Use dictionary.setdefault(key, default) to simplify.
    timeseries = timeseries or {}

    for key, value in data.items():
        if isinstance(value, dict):
            if key not in timeseries:
                timeseries[key] = {}
            timeseries[key] = value_in_embedded_dict(value, timeseries[key], time_index)
        elif time_index is None:
            if isinstance(value, Quantity):
                unit_key = (key, str(value.units))
                if unit_key not in timeseries:
                    timeseries[unit_key] = []
                timeseries[unit_key].append(value.magnitude)
            else:
                if key not in timeseries:
                    timeseries[key] = []
                timeseries[key].append(value)
        else:
            if key not in timeseries:
                timeseries[key] = {
                    'value': [],
                    'time_index': []
                }
            timeseries[key]['value'].append(value)
            timeseries[key]['time_index'].append(time_index)

    return timeseries


def get_path_list_from_dict(dictionary):
    paths_list = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            subpaths = get_path_list_from_dict(value)
            for subpath in subpaths:
                path = (key,) + subpath
                paths_list.append(path)
        else:
            path = (key,)
            paths_list.append(path)
    return paths_list


def get_value_from_path(dictionary, path):
    # noinspection PyBroadException
    try:
        return reduce(operator.getitem, path, dictionary)
    except Exception:
        traceback.print_exc()
        return None


def make_path_dict(embedded_dict):
    """ converts embedded_dict to a flat dict with path names as keys """
    path_dict = {}
    paths_list = get_path_list_from_dict(embedded_dict)
    for path in paths_list:
        path_dict[path] = get_value_from_path(embedded_dict, path)
    return path_dict
