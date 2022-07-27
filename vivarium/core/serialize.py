import math
import warnings
from typing import Any, List

import numpy as np
from pint import Unit
from pint.quantity import Quantity
from bson.codec_options import TypeEncoder, TypeRegistry, CodecOptions

from vivarium.library.units import units
from vivarium.core.registry import serializer_registry, Serializer

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

class IdentitySerializer(Serializer):  # pylint: disable=abstract-method
    '''Serializer for base types that get serialized as themselves.'''

    def __init__(self) -> None:
        super().__init__()

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


class SequenceSerializer(Serializer):  # pylint: disable=abstract-method

    def __init__(self) -> None:
        super().__init__()

    def can_deserialize(self, data: Any) -> bool:
        return isinstance(data, list)

    def deserialize(self, data: Any) -> List[Any]:
        return [deserialize_value(value) for value in data]


class DictSerializer(Serializer):  # pylint: disable=abstract-method

    def __init__(self) -> None:
        super().__init__()

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

    class Codec(TypeEncoder):
        python_type = type(units.fg)
        def transform_python(self, value):
            try:
                data = []
                for subvalue in value:
                    data.append(f"!units[{str(subvalue)}]")
                return data
            except TypeError:
                return f"!units[{str(value)}]"

    class Codec2(Codec):
        python_type = type(1*units.fg)

    def get_codecs(self):
        return [self.Codec(), self.Codec2()]

    def deserialize_from_string(self, data: str) -> Quantity:
        if data.startswith('nan'):
            unit = data[len('nan'):].strip()
            return math.nan * units(unit)
        return units(data)

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
            # The superclass deserialize() uses
            # deserialize_from_string().
            unit_data = super().deserialize(data)
            if unit is not None:
                unit_data.to(unit)
        return unit_data


class NumpySerializer(Serializer):
    """Serializer for Numpy arrays.
    Numpy array serialization is lossy--we serialize to Python lists, so
    deserialization will produce a Python list instead of a Numpy array.
    """

    def __init__(self) -> None:
        super().__init__()

    class Codec(TypeEncoder):
        python_type = np.ndarray
        def transform_python(self, value):
            return value.tolist()

    def deserialize_from_string(self, data):
        raise NotImplementedError(
            f'{self} cannot be deserialized.')

class NumpyBoolSerializer(Serializer):

    def __init__(self) -> None:
        super().__init__()

    class Codec(TypeEncoder):
        python_type = np.bool_
        def transform_python(self, value):
            return bool(value)

    def deserialize_from_string(self, data):
        raise NotImplementedError(
            f'{self} cannot be deserialized.')


class FunctionSerializer(Serializer):
    def __init__(self):
        super().__init__()

    class Codec(TypeEncoder):
        python_type = type(deserialize_value)
        def transform_python(self, value):
            return f"!FunctionSerializer[{str(value)}]"

    def deserialize_from_string(self, data):
        raise NotImplementedError(
            f'{self} cannot be deserialized.')


# Subclasses of data types handled by custom
# TypeEncoders require their own TypeEncoders.
# This includes Process, Composites, etc.
def get_codec_options():
    codecs = []
    for serializer in serializer_registry.registry.values():
        codecs += serializer.get_codecs()
    return CodecOptions(type_registry=TypeRegistry(codecs))
