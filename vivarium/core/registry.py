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

An updater function SHOULD have a name that begins with ``update_``. The
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

Each divider function SHOULD have a name prefixed with ``divide_``. The
function MUST accept a single positional argument, the value of the
variable in the mother cell. It SHOULD accept no other arguments. The
function MUST return either:

1. A :py:class:`list` with two elements: the values of the variables in
   each of the daughter cells.
2. ``None``, in which case division will be skipped for that variable.

.. note:: Dividers MAY not be deterministic and MAY not be symmetric.
    For example, a divider splitting an odd, integer-valued value may
    randomly decide which daughter cell receives the remainder.
"""
import copy
import random
import re

import numpy as np

from vivarium.library.dict_utils import deep_merge
from vivarium.library.units import Quantity


class Registry(object):
    def __init__(self):
        """A Registry holds a collection of functions or objects."""
        self.registry = {}
        self.main_keys = []

    def register(self, key, item, alternate_keys=tuple()):
        """Add an item to the registry.

        Args:
            key: Item key.
            item: The item to add.
            alternate_keys: Additional keys under which to register the
                item. These keys will not be included in the list
                returned by ``Registry.list()``.

                This may be useful if you want to be able to look up an
                item in the registry under multiple keys. For example,
                in a registry of serializers, you could register a
                serializer with its name as ``key`` and the types it can
                serialize under ``alternate_keys``. Then, when presented
                with an object to serialize, you could use the object's
                type to look up the appropriate serializer.
        """
        keys = [key]
        keys.extend(alternate_keys)
        for registry_key in keys:
            if registry_key in self.registry:
                if item != self.registry[registry_key]:
                    raise Exception('registry already contains an entry for {}: {} --> {}'.format(
                        registry_key, self.registry[key], item))
            else:
                self.registry[registry_key] = item
        self.main_keys.append(key)

    def access(self, key):
        """Get an item by key from the registry."""
        return self.registry.get(key)

    def list(self):
        return list(self.main_keys)


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
            update[k] = deep_merge(copy.deepcopy(v), new)
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


def update_dictionary(current, update):
    """Dictionary Updater
    Updater that translates _add and _delete -style updates
    into operations on a dictionary.

    Expects current to be a dictionary, with no restriction on the types of objects
    stored within it, and no defaults values.
    """
    result = current

    for key, value in update.items():
        if key == "_add":
            for added_value in value:
                added_key = added_value["key"]
                added_state = added_value["state"]
                result[added_key] = added_state
        elif key == "_delete":
            for k in value:
                del result[k]
        elif key in result:
            result[key].update(value)
        else:
            raise Exception(f"Invalid dict_value_updater key: {key}")
    return result


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


def divide_null(state):
    """Divider that causes the variable to be skipped during division.

    Returns:
        ``None`` so that no divided values are provided to the daughter
        cells. This is useful for process objects, which are handled
        separately during division.
    """
    return None


# Serializers
class Serializer:
    """Base serializer class.

    Serializers work together to convert Python objects, which may be
    collections of many different kinds of objects, into JSON-compatible
    representations. Those representations can then be deserialized to
    recover the original object.

    Most serializers convert an object to a string. To implement a
    serializer of this type, you only need to implement the
    :py:meth:`vivarium.core.registry.Serializer.serialize_to_string` and
    :py:meth:`vivarium.core.registry.Serializer.deserialize()` methods.

    Args:
        name: Name of the serializer. Defaults to the class name.
        exclusive_types: Types that are exclusively handled by this
            serializer. These can be used to quickly lookup which
            serializer handles a particular type. If an object does not
            match any of these exclusive types, the `can_serialize`
            method will be used as a fallback.

            Note that exclusive type matching is exact--inheritance is
            not accounted for. This means that if your serializer has an
            exclusive type of A and B inherits from A, an instance of B
            will not match to your serializer's exclusive types.
    """
    REGEX_FOR_NAME = re.compile('[A-Za-z0-9-_]+')
    REGEX_FOR_SERIALIZED_ANY_TYPE = re.compile(
        f'!{REGEX_FOR_NAME.pattern}\\[(.*)\\]')

    def __init__(self, name='', exclusive_types=tuple()):
        self.name = name or self.__class__.__name__
        self.regex_for_serialized = re.compile(f'!{self.name}\\[(.*)\\]')
        self.alternate_keys = (
            str(exclusive_type)
            for exclusive_type in exclusive_types
        )

    def serialize_to_string(self, data):
        """Serialize some data.

        Subclasses should override this function,

        Args:
            data: Data to serialize.

        Returns:
            The serialized data.
        """
        raise NotImplementedError(
            f'{self} has not implemented serialize_to_string().')

    def serialize(self, data):
        """Serialize some data.

        Subclasses that are serializing to strings can just implement
        :py:meth:`vivarium.core.registry.Serializer.serialize_to_string`,
        and this method will use that implementation to generate a
        string that serializes the data.

        Subclasses with more complex serialization needs must override
        this function.

        Args:
            data: Data to serialize.

        Returns:
            The serialized data.
        """
        string_serialization = self.serialize_to_string(data)
        if not isinstance(string_serialization, str):
            raise ValueError(
                f'{self}.serialize_to_string() returned invalid '
                f'serialization: {string_serialization}')

        return f'!{self.name}[{string_serialization}]'

    def can_serialize(self, data):
        """Check whether the serializer can serialize some data.

        Every subclass must implement this method.
        """
        raise NotImplementedError(
            f'{self} has not implemented can_serialize().')

    def deserialize_from_string(self, data):
        """Deserialize data from a string.

        This function is the inverse of
        :py:meth:`vivarium.core.registry.Serializer.serialize_to_string`.
        It need only be overridden in a subclass if that subclass does
        not override
        :py:meth:`vivarium.core.registry.Serializer.deserialize`.
        """
        raise NotImplementedError(
            f'{self} has not implemented deserialize_from_string().')

    def deserialize(self, data):
        """Deserialize some data.

        Subclasses relying on
        :py:meth:`vivarium.core.registry.Serializer.serialize_to_string`
        should not override this function and instead override
        :py:meth:`vivarium.core.registry.Serializer.deserialize_from_string`.

        Args:
            data: Serialized data to deserialize.

        Returns:
            The deserialized data.
        """
        string_serialization = self.regex_for_serialized.fullmatch(
            data).group(1)
        return self.deserialize_from_string(string_serialization)

    def can_deserialize(self, data):
        """Check whether the serializer can deserialize an object.

        By default, this function checks that ``data`` is a string of the
        form produced by the default implementation of
        :py:meth:`vivarium.core.registry.Serializer.serialize`. If a
        subclass overrides this ``serialize()`` method, it should also
        override ``can_deserialize()``.
        """
        if not isinstance(data, str):
            return False
        return bool(self.regex_for_serialized.fullmatch(data))
