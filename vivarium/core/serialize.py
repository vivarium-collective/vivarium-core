from collections.abc import Sequence
from typing import Any, Callable, List, Union
import warnings

import numpy as np
from bson import ObjectId
from pint import Unit
from pint.quantity import Quantity

from vivarium.core.registry import serializer_registry, Serializer
from vivarium.core.composer import Composer, Composite
from vivarium.core.process import Process
from vivarium.library.dict_utils import remove_multi_update
from vivarium.library.units import units


# Serialization and deserialization functions that handle arbitrary data
# types.


def serialize_value(value: Any) -> Any:
    # Try to lookup by exclusive type
    serializer = serializer_registry.access(str(type(value)))
    # If lookup fails, search through registered serializers
    if not serializer:
        compatible_serializers = []
        for serializer_name in serializer_registry.list():
            serializer = serializer_registry.access(serializer_name)
            if serializer.can_serialize(value):
                compatible_serializers.append(serializer)
        if not compatible_serializers:
            raise ValueError(
                f'No serializer found for {value} of type {type(value)}')
        if len(compatible_serializers) > 1:
            raise ValueError(
                f'Multiple serializers ({compatible_serializers}) found '
                f'for {value} of type {type(value)}')
        serializer = compatible_serializers[0]
    return serializer.serialize(value)


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
        super().__init__(exclusive_types=(int, float, bool, str))

    def can_serialize(self, data: Any) -> bool:
        if (
                isinstance(data, (int, float, bool, str))
                and not NumpyScalarSerializer.can_serialize(data)):
            return True
        if data is None:
            return True
        return False

    def serialize(self, data: Any) -> Any:
        return data

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


class NumpySerializer(Serializer):
    """Serializer for Numpy arrays.

    Numpy array serialization is lossy--we serialize to Python lists, so
    deserialization will produce a Python list instead of a Numpy array.
    """

    def __init__(self) -> None:
        super().__init__(exclusive_types=(np.ndarray,))

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, np.ndarray)

    def serialize(self, data: Any) -> Any:
        """Returns ``data.tolist()``."""
        lst = data.tolist()
        return serialize_value(lst)

    def can_deserialize(self, data: Any) -> bool:
        return False

    def deserialize(self, data: Any) -> None:
        raise NotImplementedError(
            'There is no deserializer for numpy arrays.')


class SequenceSerializer(Serializer):  # pylint: disable=abstract-method

    def __init__(self) -> None:
        super().__init__(exclusive_types=(list, tuple))

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, Sequence) and not isinstance(data, str)

    def serialize(self, data: Any) -> List[Any]:
        '''Serialize sequence to list of serialized elements.

        Note that sequence serialization is lossy. For example, a tuple
        will be serialized as a list, which will then be deserialized to
        a list.
        '''
        return [serialize_value(value) for value in data]

    def can_deserialize(self, data: Any) -> bool:
        return isinstance(data, list)

    def deserialize(self, data: Any) -> List[Any]:
        return [deserialize_value(value) for value in data]


class DictSerializer(Serializer):  # pylint: disable=abstract-method

    def __init__(self) -> None:
        super().__init__(exclusive_types=(dict,))

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, dict)

    def serialize(self, data: dict) -> dict:
        '''Serialize to dict of serialized elements.

        Note that dict serialization of keys is lossy. For example, a
        float key will be serialized as a string, which will then be
        deserialized to a string.
        '''
        serialized = {}
        for key, value in data.items():
            if not isinstance(key, str):
                key = str(key)
            serialized[key] = serialize_value(value)
        return serialized

    def can_deserialize(self, data: Any) -> bool:
        return isinstance(data, dict)

    def deserialize(self, data: dict) -> dict:
        return {
            key: deserialize_value(value)
            for key, value in data.items()
        }


class NumpyScalarSerializer(Serializer):
    """Serializer for Numpy scalars.

    Note that this serialization is lossy. Upon deserialization, the
    serialized values will remain Python builtin types, not Numpy
    scalars.
    """

    def __init__(self) -> None:
        super().__init__(exclusive_types=(
            np.bool_,
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64, np.float128,
            np.complex64, np.complex128, np.complex256,
        ))

    @staticmethod
    def can_serialize(data: Any) -> bool:
        return isinstance(data, (np.integer, np.floating, np.bool_))

    def serialize(
            self,
            data: Union[np.integer, np.floating, np.bool_],
            ) -> Union[int, float, bool]:
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return float(data)
        return bool(data)

    def can_deserialize(self, data: Any) -> bool:
        return False

    def deserialize(self, data: Any) -> None:
        raise NotImplementedError(
            'Deserializing serialized Numpy scalars is not supported.')


class UnitsSerializer(Serializer):
    """Serializer for data with units."""

    def __init__(self) -> None:
        super().__init__(
            name='units',
            exclusive_types=(type(units.fg), type(1 * units.fg)))

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, (Quantity, Unit))

    def serialize_to_string(self, data: Union[Quantity, Unit]) -> str:
        return str(data)

    # Here the differing argument is `unit`, which is optional, so we
    # can ignore the pylint warning.
    def serialize(  # pylint: disable=arguments-differ
            self,
            data: Union[Quantity, Unit],
            unit: Unit = None) -> Union[str, List[str]]:
        """Serialize data with units into a human-readable string.

        Args:
            data: The data to serialize. Should be a Pint object or a
                list of such objects. Note that providing a list is
                deprecated. You should use :py:func:`serialize_value`
                instead, which uses a separate list serializer.
            unit: The units to convert ``data`` to before serializing.
                Optional. If omitted, no conversion occurs. This option
                is deprecated and should not be used.

        Returns:
            The string representation of ``data`` if ``data`` is a
            single Pint object. Otherwise, a list of string
            representations.
        """
        if unit is not None:
            warnings.warn(
                'The `unit` argument to `UnitsSerializer.serialize` is '
                'deprecated.',
                DeprecationWarning,
            )

        if isinstance(data, list):
            warnings.warn(
                'Passing a list to `UnitsSerializer.serialize` is '
                'deprecated. Please use `serialize_value()` instead.',
                DeprecationWarning,
            )
            if unit is not None:
                # We can't convert `Unit` objects.
                assert isinstance(data, Quantity)
                data = [d.to(unit) for d in data]
            return [str(d) for d in data]
        if unit is not None:
            # We can't convert `Unit` objects.
            assert isinstance(data, Quantity)
            data.to(unit)
        # The superclass serialize() method uses
        # `serialize_to_string()`.
        return super().serialize(data)

    def deserialize_from_string(self, data: str) -> Quantity:
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


class ProcessSerializer(Serializer):
    """Serializer for processes.

    Currently only supports serialization (for emitting simulation
    configs).
    """

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, Process)

    def serialize_to_string(self, data: Process) -> str:
        """Create a dictionary of process name and parameters."""
        return str(dict(data.parameters, _name=data.name))

    def can_deserialize(self, data: Any) -> bool:
        return False

    def deserialize(self, data: Any) -> None:
        raise NotImplementedError(
            'Processes cannot be deserialized.')


class ComposerSerializer(Serializer):
    """Serializer for composers.

    Currently only supports serialization (for emitting simulation
    configs).
    """

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, Composer)

    def serialize_to_string(self, data: Composer) -> str:
        """Create a dictionary of composer name and parameters."""
        return str(dict(data.config, _name=str(type(data))))

    def can_deserialize(self, data: Any) -> bool:
        return False

    def deserialize(self, data: Any) -> None:
        raise NotImplementedError(
            'Composers cannot be deserialized.')


class FunctionSerializer(Serializer):
    """Serializer for functions.

    Currently only supports serialization (for emitting simulation
    configs).
    """
    def can_serialize(self, data: Any) -> bool:
        return callable(data)

    def serialize_to_string(self, data: Callable) -> str:
        return str(data)

    def can_deserialize(self, data: Any) -> bool:
        return False

    def deserialize(self, data: Any) -> None:
        raise NotImplementedError(
            'Functions cannot be deserialized.')


class ObjectIdSerializer(Serializer):
    """Serializer for BSON ObjectIds.

    Currently only supports serialization.
    """
    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, ObjectId)

    def serialize_to_string(self, data: ObjectId) -> str:
        return str(data)

    def can_deserialize(self, data: Any) -> bool:
        return False

    def deserialize(self, data: Any) -> None:
        raise NotImplementedError(
            'ObjectIds cannot be deserialized.')


def composite_specification(
        composite: Composite,
        initial_state: bool = False,
) -> dict:
    """Return the serialized specification for a :term:`composite` instance"""
    composite_dict = serialize_value(composite)
    composite_dict.pop('_schema')
    if initial_state:
        composite_initial_state = remove_multi_update(composite.initial_state())
        composite_dict = dict(
            composite_dict,
            initial_state=composite_initial_state)
    return composite_dict
