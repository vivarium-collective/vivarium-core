import math
import warnings
from typing import Any, List, Union
from collections.abc import Callable

import numpy as np
from pint import Unit
from pint.quantity import Quantity
from bson.codec_options import TypeEncoder, TypeRegistry, CodecOptions
from bson import _dict_to_bson, _bson_to_dict

from vivarium.library.units import units
from vivarium.core.registry import serializer_registry, Serializer

def serialize_value(value: Any) -> Any:
    codec_options = get_codec_options()
    value = _dict_to_bson(value, False, codec_options)
    return _bson_to_dict(value, codec_options)

# Deserialization still requires custom python code because
# BSON C extensions cannot distinguish between strings that
# should be deserialized as different types (e.g. Units)
def deserialize_value(value: Any) -> Any:
    compatible_serializers = []
    for serializer_name in serializer_registry.list():
        serializer = serializer_registry.access(serializer_name)
        if serializer.can_deserialize(value):
            compatible_serializers.append(serializer)
    if not compatible_serializers:
        raise ValueError(
            f'No deserializer found for {value}')
    if len(compatible_serializers) > 1:
        raise ValueError(
            f'Multiple deserializers ({compatible_serializers}) found '
            f'for {value}')
    serializer = compatible_serializers[0]
    return serializer.deserialize(value)


# We can ignore the abstract-method warning because Serializer only
# requires that we override serialize() or serialize_to_string(), not
# both. Similar reasoning applies to deserialize() and
# deserialize_from_string().

class IdentityDeserializer(Serializer):  # pylint: disable=abstract-method
    '''Serializer for base types that get serialized as themselves.'''

    def can_deserialize(self, data: Any) -> bool:
        if isinstance(data, (int, float, bool)):
            return True
        if isinstance(data, str):
            return not self.REGEX_FOR_SERIALIZED_ANY_TYPE.fullmatch(data)
        if data is None:
            return True
        return False

    def deserialize(self, data: Any) -> Any:
        return data


class SequenceDeserializer(Serializer):  # pylint: disable=abstract-method

    def can_deserialize(self, data: Any) -> bool:
        return isinstance(data, list)

    def deserialize(self, data: Any) -> List[Any]:
        return [deserialize_value(value) for value in data]


class DictDeserializer(Serializer):  # pylint: disable=abstract-method

    def can_deserialize(self, data: Any) -> bool:
        return isinstance(data, dict)

    def deserialize(self, data: dict) -> dict:
        return {
            key: deserialize_value(value)
            for key, value in data.items()
        }


class UnitsSerializer(Serializer):
    """Serializer for data with units."""

    def __init__(self) -> None:
        super().__init__(name='units')

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

    # Here the differing argument is `unit`, which is optional, so we
    # can ignore the pylint warning.
    def deserialize(  # pylint: disable=arguments-differ
            self, data: str, unit: Unit = None) -> Quantity:
        """Deserialize data with units from a human-readable string.

        Args:
            data: The data to deserialize. Providing a list here is
                deprecated. You should use :py:func:`deserialize_value`
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
            # The superclass deserialize() extracts ... from !units[...].
            data = super().deserialize(data)
            if data.startswith('nan'):
                unit = data[len('nan'):].strip()
                unit_data = math.nan * units(unit)
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
    class Codec(TypeEncoder):
        python_type = np.bool_
        def transform_python(self, value: np.bool_) -> bool:
            return bool(value)

    def get_codecs(self) -> List:
        return [self.Codec()]

class NumpyInt64Serializer(Serializer):
    class Codec(TypeEncoder):
        python_type = np.int64
        def transform_python(self, value: np.int64) -> int:
            return int(value)

    def get_codecs(self) -> List:
        return [self.Codec()]

class NumpyInt32Serializer(Serializer):
    class Codec(TypeEncoder):
        python_type = np.int32
        def transform_python(self, value: np.int32) -> int:
            return int(value)

    def get_codecs(self) -> List:
        return [self.Codec()]

class NumpyFloat32Serializer(Serializer):
    class Codec(TypeEncoder):
        python_type = np.float32
        def transform_python(self, value: np.float32) -> float:
            return float(value)

    def get_codecs(self) -> List:
        return [self.Codec()]

class SetSerializer(Serializer):
    class Codec(TypeEncoder):
        python_type = set
        def transform_python(self, value: set) -> List:
            return list(value)

    def get_codecs(self) -> List:
        return [self.Codec()]

class FunctionSerializer(Serializer):
    class Codec(TypeEncoder):
        python_type = type(deserialize_value)
        def transform_python(self, value: Callable) -> str:
            return f"!FunctionSerializer[{str(value)}]"

    def get_codecs(self) -> List:
        return [self.Codec()]


# Subclasses of data types handled by custom
# TypeEncoders require their own TypeEncoders.
# This includes Process, Composites, etc.
def get_codec_options() -> CodecOptions:
    codecs = []
    for serializer in serializer_registry.registry.values():
        codecs += serializer.get_codecs()
    return CodecOptions(type_registry=TypeRegistry(codecs))
