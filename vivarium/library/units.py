'''
=====
Units
=====

Here is where we create the :py:data:`units` object from which we pull
units throughout Vivarium. We also define utility functions for working
with units. Here we also import the ``_Quantity`` type as ``Quantity``,
which you can use to check if some variable has units. For example:

>>> a = 5 * units.fg
>>> b = 10
>>> isinstance(a, Quantity)
True
>>> isinstance(b, Quantity)
False
>>> a.magnitude
5
>>> isinstance(a.magnitude, Quantity)
False
'''

from __future__ import division, absolute_import, print_function

import pint
# noinspection PyProtectedMember
from pint.quantity import _Quantity as Quantity


#: Units registry that stores the units used throughout Vivarium
units = pint.UnitRegistry()

# We need to set this registry as the default application-wide so our
# registry will be used when unpickling
pint.set_application_registry(units)


def remove_units(collection):
    '''Strip the units from a collection or scalar

    If no units are present, the provided object is returned unaltered.
    Note that we assume that the provided collection consists solely of
    lists, dictionaries, Quantity objects with units to strip, and
    scalars. Behavior is undefined for other structures.

    Removal is "deep," meaning that we remove units in nested structures
    as well. For instance, given a dictionary:

    >>> d = {'a': [1, 3, {4: 3 * units.fg}, 2 * units.fg]}
    >>> remove_units(d)
    {'a': [1, 3, {4: 3}, 2]}

    Arguments:
        collection (object): The collection or scalar to strip units
            from.

    Returns:
        The collection or scalar without any units.
    '''
    if isinstance(collection, dict):
        return _remove_units_dict(collection)
    elif isinstance(collection, list):
        return _remove_units_list(collection)
    elif isinstance(collection, Quantity):
        return collection.magnitude
    return collection


def _remove_units_dict(dict_in):
    assert isinstance(dict_in, dict)
    dict_out = {}
    for key, value in dict_in.items():
        if isinstance(value, dict):
            dict_out[key] = _remove_units_dict(value)
        else:
            if isinstance(value, Quantity):
                dict_out[key] = value.magnitude
            elif isinstance(value, list):
                dict_out[key] = _remove_units_list(value)
            else:
                dict_out[key] = value
    return dict_out


def _remove_units_list(list_in):
    assert isinstance(list_in, list)
    list_out = []
    for elem in list_in:
        if isinstance(elem, list):
            removed = _remove_units_list(elem)
        else:
            if isinstance(elem, Quantity):
                removed = elem.magnitude
            elif isinstance(elem, dict):
                removed = _remove_units_dict(elem)
            else:
                removed = elem
        list_out.append(removed)
    return list_out
