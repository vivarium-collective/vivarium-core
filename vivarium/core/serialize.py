"""
==========
Serialize
==========

Collection of serializers that transform Python data into
a BSON-compatible form.
"""

import re
import math
import warnings
from typing import Any, List, Optional, Union
from collections.abc import Callable

import orjson
import numpy as np
# from pint import Unit
# try:
#     from pint.quantity import Quantity
# except ImportError:
#     from pint import Quantity
from vivarium.core.process import Process
from vivarium.library.units import units
from vivarium.core.registry import serializer_registry, Serializer


def find_numpy_and_non_strings(
    d: dict,
    curr_path: tuple=tuple(),
    saved_paths: Optional[List[tuple]]=None
) -> List[tuple]:
    """Return list of paths which terminate in a non-string or Numpy string
    dictionary key. Orjson does not handle these types of keys by default."""
    if not saved_paths:
        saved_paths = []
    if isinstance(d, dict):
        for key in d.keys():
            if not isinstance(key, str):
                saved_paths.append(curr_path + (key,))
            elif isinstance(key, np.str_):
                saved_paths.append(curr_path + (key,))
            saved_paths = find_numpy_and_non_strings(
                d[key], curr_path+(key,), saved_paths)
    return saved_paths


def serialize_value(
    value: Any,
    default: Optional[Callable]=None,
) -> Any:
    """Apply orjson-based serialization routine on ``value``.

    Args:
        value (Any): Data to be serialized. All keys must be strings. Notably,
            Numpy strings (``np.str_``) are not acceptable keys.
        default (Callable): A function that is called on any data of a type
            that is not natively supported by orjson. Returns an object that
            can be handled by default up to 254 times before an exception is
            raised.

    Returns:
        Any: Serialized data
    """
    if default is None:
        default = make_fallback_serializer_function()
    try:
        value = orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY,
            default=default)
        return orjson.loads(value)
    except TypeError as e:
        bad_keys = find_numpy_and_non_strings(value)
        raise TypeError('These paths end in incompatible non-string or Numpy '
            f'string keys: {bad_keys}').with_traceback(e.__traceback__) from e


def deserialize_value(value: Any) -> Any:
    """Find and apply the correct serializer for a value
    by calling each registered serializer's
    :py:meth:`vivarium.core.registry.Serializer.can_deserialize()`
    method. Returns the value as is if no compatible serializer
    is found.

    Args:
        value (Any): Data to be deserialized

    Raises:
        ValueError: Only one serializer should apply for any given value

    Returns:
        Any: Deserialized data
    """
    compatible_serializers = []
    for serializer_name in serializer_registry.list():
        serializer = serializer_registry.access(serializer_name)
        if serializer.can_deserialize(value):
            compatible_serializers.append(serializer)
    if not compatible_serializers:
        # Most likely a built-in type with no custom serializer/deserializer
        return value
    if len(compatible_serializers) > 1:
        raise ValueError(
            f'Multiple deserializers ({compatible_serializers}) found '
            f'for {value}')
    serializer = compatible_serializers[0]
    return serializer.deserialize(value)


class SequenceDeserializer(Serializer):
    """Iterates through lists and applies deserializers.
    """
    python_type = list

    def can_deserialize(self, data: Any) -> bool:
        return isinstance(data, list)

    def deserialize(self, data: Any) -> List[Any]:
        return [deserialize_value(value) for value in data]


class DictDeserializer(Serializer):
    """Iterates through dictionaries and applies deserializers.
    """
    python_type = dict

    def can_deserialize(self, data: Any) -> bool:
        return isinstance(data, dict)

    def deserialize(self, data: dict) -> dict:
        return {
            key: deserialize_value(value)
            for key, value in data.items()
        }


class NumpyFallbackSerializer(Serializer):
    """Orjson does not handle Numpy arrays with strings
    """
    python_type = np.ndarray

    def serialize(self, data: Any) -> list:
        return data.tolist()


class UnitsSerializer(Serializer):
    """Serializes data with units into strings of the form ``!units[...]``,
    where ``...`` is the result of calling ``str(data)``. Deserializes strings
    of this form back into data with units."""

    def __init__(self) -> None:
        super().__init__()
        self.regex_for_serialized = re.compile('!units\\[(.*)\\]')

    python_type = type(units.fg)

    def serialize(self, data: Any) -> Union[List[str], str]:
        try:
            return_value = []
            for subvalue in data:
                return_value.append(f"!units[{str(subvalue)}]")
            return return_value
        except TypeError:
            return f"!units[{str(data)}]"

    def can_deserialize(self, data: Any) -> bool:
        if not isinstance(data, str):
            return False
        return bool(self.regex_for_serialized.fullmatch(data))

    # Here the differing argument is `unit`, which is optional, so we
    # can ignore the pylint warning.
    def deserialize(self, data: str, unit=None):  # type: ignore  # pylint: disable=arguments-differ
        """Deserialize data with units from a human-readable string.

        Args:
            data: The data to deserialize. Providing a list here is
                deprecated. You should use ``deserialize_value``
                instead, which uses a separate list deserializer.
            unit: The units to convert ``data`` to after deserializing.
                If omitted, no conversion occurs. This option is
                deprecated.

        Returns:
            A single deserialized object or, if ``data`` is a list, a
            list of deserialized objects.
        """
        if unit is not None:
            warnings.warn(
                'The `unit` argument to `UnitsSerializer.deserialize` is '
                'deprecated.',
                DeprecationWarning,
            )
        if isinstance(data, list):
            warnings.warn(
                'Passing a list to `UnitsSerializer.deserialize` is '
                'deprecated. Please use `deserialize_value()` instead.',
                DeprecationWarning,
            )
            unit_data = [units(d) for d in data]
            if unit is not None:
                unit_data = [d.to(unit) for d in data]
        else:
            # Extract ... from !units[...].
            matched_regex = self.regex_for_serialized.fullmatch(data)
            if matched_regex:
                data = matched_regex.group(1)
            if data.startswith('nan'):
                unit_str = data[len('nan'):].strip()
                unit_data = math.nan * units(unit_str)
            else:
                unit_data = units(data)
            if unit is not None:
                unit_data.to(unit)  # type: ignore
        return unit_data


class QuantitySerializer(Serializer):
    """Serializes data with units into strings of the form ``!units[...]``,
    where ``...`` is the result of calling ``str(data)``. Deserializes strings
    of this form back into data with units."""
    python_type = type(1*units.fg)

    def serialize(self, data: Any) -> Union[List[str], str]:
        try:
            return_value = []
            for subvalue in data:
                return_value.append(f"!units[{str(subvalue)}]")
            return return_value
        except TypeError:
            return f"!units[{str(data)}]"


class SetSerializer(Serializer):
    """Serializer for set objects."""
    python_type = set

    def serialize(self, data: set) -> List:
        return list(data)


class FunctionSerializer(Serializer):
    """Serializer for function objects."""
    python_type = type(deserialize_value)

    def serialize(self, data: Callable) -> str:
        return f"!FunctionSerializer[{str(data)}]"


class ProcessSerializer(Serializer):
    """Serializer for processes if ``emit_process`` is enabled."""
    python_type = Process

    def serialize(self, data: Process) -> str:
        proc_str = str(dict(data.parameters, _name=data.name))
        return f"!ProcessSerializer[{proc_str}]"


def make_fallback_serializer_function() -> Callable:
    """Creates a fallback function that is called by orjson on data of
    types that are not natively supported. Define and register instances of
    :py:class:`vivarium.core.registry.Serializer()` with serialization
    routines for the types in question."""

    def default(obj: Any) -> Any:
        # Try to lookup by exclusive type
        serializer = serializer_registry.access(str(type(obj)))
        if not serializer:
            compatible_serializers = []
            for serializer_name in serializer_registry.list():
                test_serializer = serializer_registry.access(serializer_name)
                # Subclasses with registered serializers will be caught here
                if isinstance(obj, test_serializer.python_type):
                    compatible_serializers.append(test_serializer)
            if len(compatible_serializers) > 1:
                raise TypeError(
                    f'Multiple serializers ({compatible_serializers}) found '
                    f'for {obj} of type {type(obj)}')
            if not compatible_serializers:
                raise TypeError(
                    f'No serializer found for {obj} of type {type(obj)}')
            serializer = compatible_serializers[0]
            if not isinstance(obj, Process):
                # We don't warn for processes because since their types
                # based on their subclasses, it's not possible to avoid
                # searching through the serializers.
                warnings.warn(
                    f'Searched through serializers to find {serializer} '
                    f'for data of type {type(obj)}. This is '
                    f'inefficient.')
        return serializer.serialize(obj)
    return default
