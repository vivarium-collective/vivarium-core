import re
from typing import Any

import numpy as np

from vivarium.core.process import Process
from vivarium.core.serialize import serialize_value, deserialize_value
from vivarium.core.registry import Serializer
from vivarium.library.units import units


class SerializeProcess(Process):

    def ports_schema(self) -> dict:
        return {}

    def next_update(self, timestep: float, states: dict) -> dict:
        return {}


def serialize_function() -> None:
    pass


class TestSerializer(Serializer):

    def __init__(self, prefix: str = '', suffix: str = '') -> None:
        super().__init__()
        self.prefix = prefix
        self.suffix = suffix

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, str) and data.startswith('!!')

    def serialize_to_string(self, data: str) -> str:
        return f'{self.prefix}{data}{self.suffix}'

    def deserialize_from_string(self, data: str) -> str:
        if self.suffix:
            return data[len(self.prefix):-len(self.suffix)]
        return data[len(self.prefix):]


def test_serialized_in_serializer_string() -> None:
    serializer = TestSerializer(prefix='![', suffix=']')
    serialized = serializer.serialize('hi there!')
    assert serializer.deserialize(serialized) == 'hi there!'


def test_unmatched_closing_bracket_in_serializer_string() -> None:
    serializer = TestSerializer(prefix='', suffix=']')
    serialized = serializer.serialize('hi there!')
    assert serializer.deserialize(serialized) == 'hi there!'


def test_unmatched_opening_bracket_in_serializer_string() -> None:
    serializer = TestSerializer(prefix='[', suffix='')
    serialized = serializer.serialize('hi there!')
    print(serialized)
    assert serializer.deserialize(serialized) == 'hi there!'


def test_open_bracket_deep_in_serializer_string() -> None:
    serializer = TestSerializer(prefix='abc[', suffix='')
    serialized = serializer.serialize('hi there!')
    assert serializer.deserialize(serialized) == 'hi there!'


def test_close_bracket_deep_in_serializer_string() -> None:
    serializer = TestSerializer(prefix='abc]', suffix='')
    serialized = serializer.serialize('hi there!')
    assert serializer.deserialize(serialized) == 'hi there!'


def test_serialized_prefixing_serializer_string() -> None:
    serializer = TestSerializer(prefix='!TestSerializer[test]', suffix='')
    serialized = serializer.serialize('hi there!')
    assert serializer.deserialize(serialized) == 'hi there!'


def test_exclamation_point_prefixing_serializer_string() -> None:
    serializer = TestSerializer(prefix='!', suffix='')
    serialized = serializer.serialize('hi there!')
    assert serializer.deserialize(serialized) == 'hi there!'


def test_exclamation_point_suffixing_serializer_string() -> None:
    serializer = TestSerializer(prefix='', suffix='!')
    serialized = serializer.serialize('hi there!')
    assert serializer.deserialize(serialized) == 'hi there!'


def test_serialization_full() -> None:
    to_serialize = {
        'process': SerializeProcess(),
        1: True,
        'numpy_int': np.array([1, 2, 3]),
        'numpy_float': np.array([1.1, 2.2, 3.3]),
        'numpy_str': np.array(['a', 'b', 'c']),
        'numpy_bool': np.array([True, True, False]),
        'numpy_matrix': np.array([[1, 2], [3, 4]]),
        'list_units': [1 * units.fg, 2 * units.fg],
        'list': [True, False, 'test', 1, None],
        'quantity': 5 * units.fg,
        'unit': units.fg,
        'dict': {
            'a': False,
        },
        'function': serialize_function,
    }
    serialized = serialize_value(to_serialize)
    assert re.fullmatch(
        '!FunctionSerializer\\[<function serialize_function at 0x[0-9a-f]+>\\]',
        serialized.pop('function'))
    expected_serialized = {
        'process': (
            "!ProcessSerializer[{'time_step': 1.0, '_name': "
            "'SerializeProcess'}]"
        ),
        '1': True,
        'numpy_int': [1, 2, 3],
        'numpy_float': [1.1, 2.2, 3.3],
        'numpy_str': ['a', 'b', 'c'],
        'numpy_bool': [True, True, False],
        'numpy_matrix': [[1, 2], [3, 4]],
        'list_units': [
            '!units[1 femtogram]',
            '!units[2 femtogram]',
        ],
        'list': [True, False, 'test', 1, None],
        'quantity': '!units[5 femtogram]',
        'unit': '!units[femtogram]',
        'dict': {
            'a': False,
        },
    }
    assert serialized == expected_serialized

    # Processes cannot be deserialized.
    expected_serialized.pop('process')

    deserialized = deserialize_value(expected_serialized)
    expected_deserialized = {
        '1': True,
        'numpy_int': [1, 2, 3],
        'numpy_float': [1.1, 2.2, 3.3],
        'numpy_str': ['a', 'b', 'c'],
        'numpy_bool': [True, True, False],
        'numpy_matrix': [[1, 2], [3, 4]],
        'list_units': [1 * units.fg, 2 * units.fg],
        'list': [True, False, 'test', 1, None],
        'quantity': 5 * units.fg,
        'unit': 1 * units.fg,
        'dict': {
            'a': False,
        },
    }
    assert deserialized == expected_deserialized
