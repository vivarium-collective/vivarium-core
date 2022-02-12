import re

import numpy as np

from vivarium.core.process import Process
from vivarium.core.serialize import serialize_value, deserialize_value
from vivarium.library.units import units


class SerializeProcess(Process):

    def ports_schema(self):
        return {}

    def next_update(self, timestep, states):
        return {}


def serialize_function():
    pass


def test_serialization():
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
            '!UnitsSerializer[1 femtogram]',
            '!UnitsSerializer[2 femtogram]',
        ],
        'list': [True, False, 'test', 1, None],
        'quantity': '!UnitsSerializer[5 femtogram]',
        'unit': '!UnitsSerializer[femtogram]',
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
