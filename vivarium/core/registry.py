"""
==============================================
Registry of Updaters, Dividers, and Serializers
==============================================

You should interpret words and phrases that appear fully capitalized in
this document as described in :rfc:`2119`. Here is a brief summary of
the RFC:

* "MUST" indicates absolute requirements. Vivarium may not work
  correctly if you don't follow these.
* "SHOULD" indicates strong suggestions. You might have a valid reason
  for deviating from them, but be careful that you understand the
  ramifications.
* "MAY" indicates truly optional features that you can include or
  exclude as you wish.

--------
Updaters
--------

Each :term:`updater` is defined as a function whose name begins with
``update_``. Vivarium uses these functions to apply :term:`updates` to
:term:`variables`. Updater names are registered in
:py:data:`updater_registry`, which maps these names to updater functions.

Updater API
===========

An updater function MUST have a name that begins with ``update_``. The
function MUST accept exactly three positional arguments: the first MUST
be the current value of the variable (i.e. before applying the update),
the second MUST be the value associated with the variable in the update,
and the third MUST be either a dictionary of states from the simulation
hierarchy or None if no ``port_mapping`` key was specified in the
updater definition. The function SHOULD not accept any other parameters.
The function MUST return the updated value of the variable only.

--------
Dividers
--------

Each :term:`divider` is defined by a function that follows the API we
describe below. Vivarium uses these dividers to generate daughter cell
states from the mother cell's state. Divider names are registered in
:py:data:`divider_registry`, which maps these names to divider functions.

Divider API
===========

Each divider function MUST have a name prefixed with ``_divide``. The
function MUST accept a single positional argument, the value of the
variable in the mother cell. It SHOULD accept no other arguments. The
function MUST return a :py:class:`list` with two elements: the values of
the variables in each of the daughter cells.

.. note:: Dividers MAY not be deterministic and MAY not be symmetric.
    For example, a divider splitting an odd, integer-valued value may
    randomly decide which daughter cell receives the remainder.

--------
Derivers
--------

Each :term:`deriver` is defined as a separate :term:`process`, but here
deriver names are mapped to processes by :py:data:`deriver_registry`. The
available derivers are:

* **mmol_to_counts**: :py:class:`vivarium.processes.derive_counts.DeriveCounts`
* **counts_to_mmol**:
  :py:class:`vivarium.processes.derive_concentrations.DeriveConcentrations`
* **mass**: :py:class:`vivarium.processes.tree_mass.TreeMass`
* **globals**:
  :py:class:`vivarium.processes.derive_globals.DeriveGlobals`

See the documentation for each :term:`process class` for more details on
that deriver.
"""


from __future__ import absolute_import, division, print_function

import random

import numpy as np

from vivarium.library.dict_utils import deep_merge
from vivarium.library.units import Quantity, units


class Registry(object):
    def __init__(self):
        self.registry = {}

    def register(self, key, item):
        if key in self.registry:
            if item != self.registry[key]:
                raise Exception('registry already contains an entry for {}: {} --> {}'.format(key, self.registry[key], item))
        else:
            self.registry[key] = item

    def access(self, key):
        return self.registry.get(key)


#: Maps process names to :term:`process classes`
process_registry = Registry()


## updater functions

def update_merge(current_value, new_value, states):
    """Merge Updater

    Returns:
        dict: The merger of ``current_value`` and ``new_value``. For any
        shared keys, the value in ``new_value`` is used.
    """
    update = current_value.copy()
    for k, v in current_value.items():
        new = new_value.get(k)
        if isinstance(new, dict):
            update[k] = deep_merge(dict(v), new)
        else:
            update[k] = new
    return update

def update_set(current_value, new_value, states):
    """Set Updater

    Returns:
        The value provided in ``new_value``.
    """
    return new_value

def update_accumulate(current_value, new_value, states):
    """Accumulate Updater

    Returns:
        The sum of ``current_value`` and ``new_value``.
    """
    return current_value + new_value


#: Maps updater names to updater functions
updater_registry = Registry()
updater_registry.register('accumulate', update_accumulate)
updater_registry.register('set', update_set)
updater_registry.register('merge', update_merge)


## divider functions
def divide_set(state):
    """Set Divider

    Returns:
        A list ``[state, state]``. No copying is performed.
    """
    return [state, state]

def divide_split(state):
    """Split Divider

    Arguments:
        state: Must be an :py:class:`int`, a :py:class:`float`, or a
            :py:class:`str` of value ``Infinity``.

    Returns:
        A list, each of whose elements contains half of ``state``. If
        ``state`` is an :py:class:`int`, the remainder is placed at
        random in one of the two elements. If ``state`` is infinite, the
        return value is ``[state, state]`` (no copying is done).

    Raises:
        Exception: if ``state`` is of an unrecognized type.
    """
    if isinstance(state, (int, np.integer)):
        remainder = state % 2
        half = int(state / 2)
        if random.choice([True, False]):
            return [half + remainder, half]
        else:
            return [half, half + remainder]
    elif state == float('inf') or state == 'Infinity':
        # some concentrations are considered infinite in the environment
        # an alternative option is to not divide the local environment state
        return [state, state]
    elif isinstance(state, (float, Quantity)):
        half = state/2
        return [half, half]
    else:
        raise Exception('can not divide state {} of type {}'.format(state, type(state)))

def divide_zero(state):
    """Zero Divider

    Returns:
        ``[0, 0]`` regardless of input
    """
    return [0, 0]

def divide_split_dict(state):
    """Split-Dictionary Divider

    Returns:
        A list of two dictionaries. The first dictionary stores the
        first half of the key-value pairs in ``state``, and the second
        dictionary stores the rest of the key-value pairs.

        .. note:: Since dictionaries are unordered, you should avoid
            making any assumptions about which keys will be sent to
            which daughter cell.
    """
    if state is None:
        state = {}
    d1 = dict(list(state.items())[len(state) // 2:])
    d2 = dict(list(state.items())[:len(state) // 2])
    return [d1, d2]

def assert_no_divide(state):
    '''Assert that the variable is never divided

    Raises:
        AssertionError: If the variable is divided
    '''
    raise AssertionError('Variable cannot be divided')


#: Map divider names to divider functions
divider_registry = Registry()
divider_registry.register('set', divide_set)
divider_registry.register('split', divide_split)
divider_registry.register('split_dict', divide_split_dict)
divider_registry.register('zero', divide_zero)
divider_registry.register('zero', divide_zero)
divider_registry.register('no_divide', assert_no_divide)


# Serializers
class Serializer(object):
    def serialize(self, data):
        return data

    def deserialize(self, data):
        return data

class NumpySerializer(Serializer):
    def serialize(self, data):
        return data.tolist()

    def deserialize(self, data):
        return np.array(data)

class NumpyScalarSerializer(Serializer):
    def serialize(self, data):
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return float(data)
        raise ValueError(
            'Cannot serialize numpy scalar {} of type {}.'.format(
                data, type(data)
            )
        )

    def deserialize(self, data):
        if isinstance(data, int):
            return np.int64(data)
        if isinstance(data, float):
            return np.float64(data)
        raise ValueError(
            'Cannot deserialize scalar {} of type {}.'.format(
                data, type(data)
            )
        )

class UnitsSerializer(Serializer):
    def serialize(self, data):
        return data.magnitude

class ProcessSerializer(Serializer):
    def serialize(self, data):
        return dict(data.parameters, _name = data.name)

class GeneratorSerializer(Serializer):
    def serialize(self, data):
        return dict(data.config, _name = str(type(data)))

class FunctionSerializer(Serializer):
    def serialize(self, data):
        return str(data)

# register serializers in the serializer_registry
serializer_registry = Registry()
serializer_registry.register('numpy', NumpySerializer())
serializer_registry.register('numpy_scalar', NumpyScalarSerializer())
serializer_registry.register('units', UnitsSerializer())
serializer_registry.register('process', ProcessSerializer())
serializer_registry.register('compartment', GeneratorSerializer())
serializer_registry.register('function', FunctionSerializer())
