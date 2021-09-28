"""
===============================================
Registry of Updaters, Dividers, and Serializers
===============================================

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
"""

import random

import numpy as np

from vivarium.library.dict_utils import deep_merge
from vivarium.library.units import Quantity, units


class Registry(object):
    def __init__(self):
        """A Registry holds a collection of functions or objects."""
        self.registry = {}

    def register(self, key, item):
        """Add an item to the registry.

        Args:
            key: Item key.
            item: The item to add.
        """
        if key in self.registry:
            if item != self.registry[key]:
                raise Exception('registry already contains an entry for {}: {} --> {}'.format(
                    key, self.registry[key], item))
        else:
            self.registry[key] = item

    def access(self, key):
        """Get an item by key from the registry."""
        return self.registry.get(key)

    def list(self):
        return list(self.registry.keys())


# Initialize registries
# These are imported into module __init__.py files,
# where the functions and classes are registered upon import

#: Maps process names to :term:`process classes`
process_registry = Registry()

#: Map updater names to :term:`updater` functions
updater_registry = Registry()

#: Map divider names to :term:`divider` functions
divider_registry = Registry()

#: Map serializer names to :term:`serializer` classes
serializer_registry = Registry()

#: Map serializer names to :term:`Emitter` classes
emitter_registry = Registry()


# Updaters, Dividers, and Serializers
# These can be defined here, but are registered in the base module's __init__.py file

# Updater functions
def update_merge(current_value, new_value):
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


def update_set(current_value, new_value):
    """Set Updater

    Returns:
        The value provided in ``new_value``.
    """
    return new_value


def update_null(current_value, new_value):
    """Null Updater

    Returns:
        The value provided in ``current_value``.
    """
    return current_value


def update_accumulate(current_value, new_value):
    """Accumulate Updater

    Returns:
        The sum of ``current_value`` and ``new_value``.
    """
    return current_value + new_value


def update_nonnegative_accumulate(current_value, new_value):
    """Non-negative Accumulate Updater

    Returns:
        The sum of ``current_value`` and ``new_value`` if positive, 0 if negative.
    """
    updated_value = current_value + new_value
    if isinstance(updated_value, np.ndarray):
        updated_value[updated_value < 0] = 0
        return updated_value
    elif updated_value >= 0:
        return updated_value
    else:
        return 0 * updated_value


# Divider functions
def divide_set(state):
    """Set Divider

    Returns:
        A list ``[state, state]``. No copying is performed.
    """
    return [state, state]


def divide_set_value(state, config):
    """Set Value Divider
    Args:
        'state': value
    Returns:
        A list ``[value, value]``. No copying is performed.
    """
    value = config['value']
    return [value, value]


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


def divide_binomial(state):
    """Binomial Divider
    """
    counts_1 = np.random.binomial(state, 0.5)
    counts_2 = state - counts_1
    return [counts_1, counts_2]


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
    """Assert that the variable is never divided

    Raises:
        AssertionError: If the variable is divided
    """
    raise AssertionError('Variable cannot be divided')


# Serializers
class Serializer(object):
    """Base serializer class."""
    def serialize(self, data):
        """Serialize some data.

        Subclasses should override this function, which currently
        returns the data unaltered.

        Args:
            data: Data to serialize.

        Returns:
            The serialized data.
        """
        return data

    def deserialize(self, data):
        """Deserialize some data.

        Subclasses should override this function, which currently
        returns the data unaltered.

        Args:
            data: Serialized data to deserialize.

        Returns:
            The deserialized data.
        """
        return data


class NumpySerializer(Serializer):
    """Serializer for Numpy arrays."""

    def serialize(self, data):
        """Returns ``data.tolist()``."""
        return data.tolist()

    def deserialize(self, data):
        """Passes ``data`` to ``np.array``."""
        return np.array(data)


class NumpyScalarSerializer(Serializer):
    """Serializer for Numpy scalars."""

    def serialize(self, data):
        """Convert scalar to :py:class:`int` or :py:class:`float`."""
        if isinstance(data, (int, np.integer)):
            return int(data)
        if isinstance(data, (float, np.floating)):
            return float(data)
        raise ValueError(
            'Cannot serialize numpy scalar {} of type {}.'.format(
                data, type(data)
            )
        )

    def deserialize(self, data):
        """Convert to ``np.int64`` or ``np.float64``."""
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
    """Serializer for data with units."""

    def serialize(self, data, unit=None):
        """Serialize data with units into a human-readable string.

        Args:
            data: The data to serialize. Should be a Pint object or a
                list of such objects.
            unit: The units to convert ``data`` to before serializing.
                Optional. If omitted, no conversion occurs.

        Returns:
            The string representation of ``data`` if ``data`` is a
            single Pint object. Otherwise, a list of string
            representations.
        """
        if isinstance(data, list):
            if unit is not None:
                data = [d.to(unit) for d in data]
            return [str(d) for d in data]
        else:
            if unit is not None:
                data.to(unit)
            return str(data)

    def deserialize(self, data, unit=None):
        """Deserialize data with units from a human-readable string.

        Args:
            data: The data to deserialize.
            unit: The units to convert ``data`` to after deserializing.
                If omitted, no conversion occurs.

        Returns:
            A single deserialized object or, if ``data`` is a list, a
            list of deserialized objects.
        """
        if isinstance(data, list):
            unit_data = [units(d) for d in data]
            if unit is not None:
                unit_data = [d.to(unit) for d in data]
        else:
            unit_data = units(data)
            if unit is not None:
                unit_data.to(unit)
        return unit_data


class ProcessSerializer(Serializer):
    """Serializer for processes.

    Currently only supports serialization (for emtting simulation
    configs).
    """
    def serialize(self, data):
        """Create a dictionary of process name and parameters."""
        return dict(data.parameters, _name=data.name)


class ComposerSerializer(Serializer):
    """Serializer for composers.

    Currently only supports serialization (for emtting simulation
    configs).
    """

    def serialize(self, data):
        """Create a dictionary of composer name and parameters."""
        return dict(data.config, _name=str(type(data)))


class FunctionSerializer(Serializer):
    """Serializer for functions.

    Currently only supports serialization (for emtting simulation
    configs).
    """

    def serialize(self, data):
        """Call ``data.__str__()``."""
        return str(data)
