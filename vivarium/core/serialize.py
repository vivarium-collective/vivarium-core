from typing import Any, cast, Callable, Dict

import numpy as np
from bson import ObjectId
from pint import Unit, UndefinedUnitError
from pint.quantity import Quantity

from vivarium import serializer_registry
from vivarium.core.composer import Composer
from vivarium.core.process import Process


def serialize_value(value: Any) -> Any:
    """Attempt to serialize a value.

    For this function, consider "serializable" to mean serializiable by
    this function.  This function can serialize the following kinds of
    values:

    * :py:class:`dict` whose keys implement ``__str__`` and whose values
      are serializable. Keys are serialized by calling ``__str__`` and
      values are serialized by this function.
    * :py:class:`list` and :py:class:`tuple`, whose values are
      serializable. The value is serialized as a list of the serialized
      values.
    * Numpy ndarray objects are handled by
      :py:class:`vivarium.core.registry.NumpySerializer`.
    * :py:class:`pint.Quantity` objects are handled by
      :py:class:`vivarium.core.registry.UnitsSerializer`.
    * Functions are handled by
      :py:class:`vivarium.core.registry.FunctionSerializer`.
    * :py:class:`vivarium.core.process.Process` objects are handled by
      :py:class:`vivarium.core.registry.ProcessSerializer`.
    * :py:class:`vivarium.core.process.Composer` objects are handled by
      :py:class:`vivarium.core.registry.ComposerSerializer`.
    * Numpy scalars are handled by
      :py:class:`vivarium.core.registry.NumpyScalarSerializer`.
    * ``ObjectId`` objects are serialized by calling its ``__str__``
      function.

    When provided with a serializable value, the returned serialized
    value is suitable for inclusion in a JSON object.

    Args:
        value: The value to serialize.

    Returns:
        The serialized value if ``value`` is serializable. Otherwise,
        ``value`` is returned unaltered.
    """
    if isinstance(value, dict):
        value = cast(dict, value)
        return _serialize_dictionary(value)
    if isinstance(value, list):
        value = cast(list, value)
        return _serialize_list(value)
    if isinstance(value, tuple):
        value = cast(tuple, value)
        return _serialize_list(list(value))
    if isinstance(value, np.ndarray):
        value = cast(np.ndarray, value)
        return serializer_registry.access('numpy').serialize(value)
    if isinstance(value, Quantity):
        value = cast(Quantity, value)
        return serializer_registry.access('units').serialize(value)
    if isinstance(value, Unit):
        value = cast(Unit, value)
        return serializer_registry.access('units').serialize(value)
    if callable(value):
        value = cast(Callable, value)
        return serializer_registry.access('function').serialize(value)
    if isinstance(value, Process):
        value = cast(Process, value)
        return _serialize_dictionary(
            serializer_registry.access('process').serialize(value))
    if isinstance(value, Composer):
        value = cast(Composer, value)
        return _serialize_dictionary(
            serializer_registry.access('composer').serialize(value))
    if isinstance(value, (np.integer, np.floating)):
        return serializer_registry.access(
            'numpy_scalar').serialize(value)
    if isinstance(value, ObjectId):
        value = cast(ObjectId, value)
        return str(value)
    return value


def deserialize_value(value: Any) -> Any:
    """Attempt to deserialize a value.

    Supports deserializing the following kinds ov values:

    * :py:class:`dict` with serialized values and keys that need not be
      deserialized. The values will be deserialized with this function.
    * :py:class:`list` of serialized values. The values will be
      deserialized with this function.
    * :py:class:`str` which are serialized :py:class:`pint.Quantity`
      values.  These will be deserialized with
      :py:class:`vivarium.core.registry.UnitsSerializer`.

    Args:
        value: The value to deserialize.

    Returns:
        The deserialized value if ``value`` is of a supported type.
        Otherwise, returns ``value`` unmodified.
    """
    if isinstance(value, dict):
        value = cast(dict, value)
        return _deserialize_dictionary(value)
    if isinstance(value, list):
        value = cast(list, value)
        return _deserialize_list(value)
    if isinstance(value, str):
        value = cast(str, value)
        try:
            return serializer_registry.access(
                'units').deserialize(value)
        except UndefinedUnitError:
            return value
    return value


def _serialize_list(lst: list) -> list:
    serialized = []
    for value in lst:
        serialized.append(serialize_value(value))
    return serialized


def _serialize_dictionary(d: dict) -> Dict[str, Any]:
    serialized = {}
    for key, value in d.items():
        if not isinstance(key, str):
            key = str(key)
        serialized[key] = serialize_value(value)
    return serialized


def _deserialize_list(lst: list) -> list:
    deserialized = []
    for value in lst:
        deserialized.append(deserialize_value(value))
    return deserialized


def _deserialize_dictionary(d: dict) -> dict:
    deserialized = {}
    for key, value in d.items():
        deserialized[key] = deserialize_value(value)
    return deserialized
