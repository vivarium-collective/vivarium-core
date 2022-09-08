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
from typing import Any, List, Union
from collections.abc import Callable

import numpy as np
from pint import Unit
from pint.quantity import Quantity
from bson.codec_options import TypeEncoder, TypeRegistry, CodecOptions
from bson import _dict_to_bson, _bson_to_dict
from vivarium.core.process import Process

from vivarium.library.units import units
from vivarium.core.registry import serializer_registry, Serializer

def serialize_value(
    value: Any,
    codec_options: CodecOptions=None
) -> Any:
    """Apply PyMongo's TypeCodec-based serialization routine on ``value``.

    Args:
        value (Any): Data to be serialized
        codec_options (bson.codec_options.CodecOptions): Options used when
            encoding / decoding BSON. Defaults to None, in which case options
            are generated from the currently registered serializer codecs.

    Returns:
        Any: Serialized data
    """
    if not codec_options:
        codec_options = get_codec_options()
    value = _dict_to_bson(value, False, codec_options)
    return _bson_to_dict(value, codec_options)

# Deserialization still requires custom python code because
# BSON C extensions cannot distinguish between strings that
# should be deserialized as different types (e.g. Units)
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


class SequenceDeserializer(Serializer):  # pylint: disable=abstract-method
    """Iterates through lists and applies deserializers.
    """
    def can_deserialize(self, data: Any) -> bool:
        return isinstance(data, list)

    def deserialize(self, data: Any) -> List[Any]:
        return [deserialize_value(value) for value in data]


class DictDeserializer(Serializer):  # pylint: disable=abstract-method
    """Iterates through dictionaries and applies deserializers.
    """
    def can_deserialize(self, data: Any) -> bool:
        return isinstance(data, dict)

    def deserialize(self, data: dict) -> dict:
        return {
            key: deserialize_value(value)
            for key, value in data.items()
        }


class UnitsSerializer(Serializer):
    """Serializes data with units into strings of the form ``!units[...]``,
    where ``...`` is the result of calling ``str(data)``. Deserializes strings
    of this form back into data with units."""

    def __init__(self) -> None:
        super().__init__(name='units')
        self.regex_for_serialized = re.compile(f'!{self.name}\\[(.*)\\]')

    class UnitCodec(TypeEncoder):
        python_type = type(units.fg)
        def transform_python(self, value: Any) -> Union[List[str], str]:
            try:
                data = []
                for subvalue in value:
                    data.append(f"!units[{str(subvalue)}]")
                return data
            except TypeError:
                return f"!units[{str(value)}]"

    class QuantityCodec(UnitCodec):
        python_type = type(1*units.fg)

    def get_codecs(self) -> List:
        return [self.UnitCodec(), self.QuantityCodec()]

    def serialize(self, data: Any) -> str:
        for codec in self.codecs:
            if isinstance(data, codec.python_type):
                return codec.transform_python(data)
        raise TypeError(f'{data} is not of type Unit or Quantity')

    def can_deserialize(self, data: Any) -> bool:
        if not isinstance(data, str):
            return False
        return bool(self.regex_for_serialized.fullmatch(data))

    # Here the differing argument is `unit`, which is optional, so we
    # can ignore the pylint warning.
    def deserialize(  # pylint: disable=arguments-differ
            self, data: str, unit: Unit = None) -> Quantity:
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
                unit_data.to(unit)
        return unit_data


class NumpySerializer(Serializer):
    """Serializer for Numpy arrays.
    Numpy array serialization is lossy--we serialize to Python lists, so
    deserialization will produce a Python list instead of a Numpy array.
    """

    class Codec(TypeEncoder):
        python_type = np.ndarray
        def transform_python(self, value: np.ndarray) -> List:
            return value.tolist()

    def get_codecs(self) -> List:
        return [self.Codec()]

class NumpyBoolSerializer(Serializer):
    """Serializer for ``np.bool_`` objects."""
    class Codec(TypeEncoder):
        python_type = np.bool_
        def transform_python(self, value: np.bool_) -> bool:
            return bool(value)

    def get_codecs(self) -> List:
        return [self.Codec()]

class NumpyInt64Serializer(Serializer):
    """Serializer for ``np.int64`` objects."""
    class Codec(TypeEncoder):
        python_type = np.int64
        def transform_python(self, value: np.int64) -> int:
            return int(value)

    def get_codecs(self) -> List:
        return [self.Codec()]

class NumpyInt32Serializer(Serializer):
    """Serializer for ``np.int32`` objects."""
    class Codec(TypeEncoder):
        python_type = np.int32
        def transform_python(self, value: np.int32) -> int:
            return int(value)

    def get_codecs(self) -> List:
        return [self.Codec()]

class NumpyFloat32Serializer(Serializer):
    """Serializer for ``np.float32`` objects."""
    class Codec(TypeEncoder):
        python_type = np.float32
        def transform_python(self, value: np.float32) -> float:
            return float(value)

    def get_codecs(self) -> List:
        return [self.Codec()]

class SetSerializer(Serializer):
    """Serializer for set objects."""
    class Codec(TypeEncoder):
        python_type = set
        def transform_python(self, value: set) -> List:
            return list(value)

    def get_codecs(self) -> List:
        return [self.Codec()]

class FunctionSerializer(Serializer):
    """Serializer for function objects."""
    class Codec(TypeEncoder):
        python_type = type(deserialize_value)
        def transform_python(self, value: Callable) -> str:
            return f"!FunctionSerializer[{str(value)}]"

    def get_codecs(self) -> List:
        return [self.Codec()]

class ProcessSerializer(Serializer):
    """Serializer for processes if ``emit_process`` is enabled."""

    def __init__(self) -> None:
        super().__init__(name='processes')

    def serialize(self, data: Process) -> str:
        proc_str = str(dict(data.parameters, _name=data.name))
        return f"!ProcessSerializer[{proc_str}]"

# Subclasses of data types handled by custom
# TypeEncoders require their own TypeEncoders.
# This includes Process, Composites, etc.
def get_codec_options() -> CodecOptions:
    """Returns a set of options used for serializing and
    deserializing BSON by collecting codecs from all registered
    serializers."""
    codecs = []
    for serializer in serializer_registry.registry.values():
        codecs += serializer.get_codecs()
    return CodecOptions(type_registry=TypeRegistry(codecs))
