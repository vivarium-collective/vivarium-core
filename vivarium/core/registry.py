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
    
-----------
Serializers
-----------

Each :term:`serializer` is defined as a class that follows the API we
describe below. Vivarium uses these serializers to convert emitted data
into a BSON-compatible format for database storage. Serializer names are
registered in :py:data:`serializer_registry`, which maps these names to
serializer subclasses.

Serializer API
==============

Each serializer SHOULD implement the following methods and attributes:

1. One or more codecs in of the type :py:class:`bson.codec_options.TypeCodec`
   or one of its subclasses
2. The :py:meth:`vivarium.core.registry.Serializer.get_codecs()` method
3. The :py:meth:`vivarium.core.registry.Serializer.deserialize_from_string()`
   method OR the :py:meth:`vivarium.core.registry.Serializer.deserialize()`
   and :py:meth:`vivarium.core.registry.Serializer.can_deserialize()` methods
"""
import copy
import random
import re

import numpy as np
from bson.codec_options import TypeCodec

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
                item in the registry under multiple keys.
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


class Dummy():
    """Dummy class used to define example codec.
    """
    dummy = ''

# Serializers
class Serializer:
    """Base serializer class.

    Serializers work together to convert Python objects, which may be
    collections of many different kinds of objects, into BSON-compatible
    representations. Those representations can then be deserialized to
    recover the original object.
    
    Serialization is mostly handled via the 
    :py:class:`bson.codec_options.TypeCodec` interface of PyMongo.
    If a store is assigned a custom serializer using the
    ``_serializer`` key, serialization occurs instead via the
    :py:meth:`vivarium.core.registry.Serializer.serialize()` method and
    may be slower.
    
    Deserialization is typically handled by either the
    :py:meth:`vivarium.core.registry.Serializer.deserialize()` or
    :py:meth:`vivarium.core.registry.Serializer.deserialize_from_string()`
    method depending on whether the subclass has a one-to-one mapping
    between BSON and Python types (no codecs in the subclass or any other
    subclass share the same input or output types as the codecs in the
    subclass). Alternatively, if the defined codecs define ``bson_type``
    and the ``transform_bson()`` method, deserialization is handled by
    PyMongo, and :py:meth:`vivarium.core.registry.Serializer.deserialize()`
    can be safely overriden to simply ``pass``.

    Args:
        name: Name of the serializer. Defaults to the class name.
    """
    REGEX_FOR_NAME = re.compile('[A-Za-z0-9-_]+')
    REGEX_FOR_SERIALIZED_ANY_TYPE = re.compile(
        f'!{REGEX_FOR_NAME.pattern}\\[(.*)\\]')

    def __init__(self, name=''):
        self.name = name or self.__class__.__name__
        self.regex_for_serialized = re.compile(f'!{self.name}\\[(.*)\\]')
        self.codecs = self.get_codecs()

    class Codec(TypeCodec):
        """Example TypeCodec.
        
        All subclasses must have one or more codecs. Note that type checking
        is exact, meaning subclasses of defined types require their own codecs.
        """
        python_type = type(Dummy()) #: Python type to serialize
        def transform_python(self, value):
            """Converts ``value`` with type ``python_type`` into a
            BSON-compatible type.
            """
            return None
        bson_type = type(Dummy()) #: BSON type to deserialize
        def transform_bson(self, value):
            """Converts ``value`` with type ``bson_type`` into a
            Python type."""
            return None
    
    def get_codecs(self):
        """Get list of codecs in serializer.
        
        All subclasses should override this method.
        """
        return [self.Codec()]
    
    def serialize(self, data):
        """Serialize data using correct codec.
        
        This is typically only called in the case that individual stores
        are assigned custom serializers. For maximum efficiency, serialization
        should be left to ``pymongo`` instead of calling this function.
        
        Do not override this function without good reason.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized data
        """
        for codec in self.codecs:
            if isinstance(data, codec.python_type):
                return codec.transform_python(data)

    def deserialize_from_string(self, data):
        """Deserialize data from a string.

        This function should only be overwritten if a codec does not form
        a one-to-one mapping between BSON and Python types and instead
        serializes data to strings of the form::

            f'!{self.name or self.__class__.__name__}[serialized_data]'

        It should turn ``serialized_data`` into the correct Python type.
        
        Args:
            data: Serialized data without the 
                ``!{self.name or self.__class__.__name__}``
                prefix or the brackets
                
        Returns:
            The deserialized data
        """
        raise NotImplementedError(
            f'{self} has not implemented deserialize_from_string().')

    def deserialize(self, data):
        """Deserialize some data.

        This function should typically only be overwritten if all codecs in
        the subclass form one-to-one mappings between BSON and Python types.
        Otherwise, leave this function alone and only override
        :py:meth:`vivarium.core.registry.Serializer.deserialize_from_string`.

        Args:
            data: Serialized data to deserialize

        Returns:
            The deserialized data
        """
        string_serialization = self.regex_for_serialized.fullmatch(
            data).group(1)
        return self.deserialize_from_string(string_serialization)

    def can_deserialize(self, data):
        """Check whether the serializer can deserialize an object.

        By default, this function checks that ``data`` is a string of the
        form::
        
            f'!{self.name or self.__class__.__name__}[serialized_data]'
        
        If a subclass does not serialize to this format, it should also
        override ``can_deserialize()``.
        """
        if not isinstance(data, str):
            return False
        return bool(self.regex_for_serialized.fullmatch(data))
