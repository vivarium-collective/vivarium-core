from vivarium.core.registry import (
    process_registry,
    updater_registry,
    divider_registry,
    serializer_registry,
)

# processes
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.tree_mass import TreeMass
from vivarium.processes.derive_concentrations import DeriveConcentrations
from vivarium.processes.derive_counts import DeriveCounts
from vivarium.processes.timeline import TimelineProcess
from vivarium.processes.nonspatial_environment import NonSpatialEnvironment


# register processes
process_registry.register(MetaDivision.name, MetaDivision)
process_registry.register(TreeMass.name, TreeMass)
process_registry.register(DeriveConcentrations.name, DeriveConcentrations)
process_registry.register(DeriveCounts.name, DeriveCounts)
process_registry.register(TimelineProcess.name, TimelineProcess)
process_registry.register(NonSpatialEnvironment.name, NonSpatialEnvironment)


# updater functions
def update_merge(current_value, new_value, states):
    """Merge Updater

    Returns:
        dict: The merger of ``current_value`` and ``new_value``. For any
        shared keys, the value in ``new_value`` is used.
    """
    update = current_value.copy()
    for k, v in current_value.items():
        new = new_value.get(k)
        if isinstance(new, dict):
            update[k] = deep_merge(dict(v), new)
        else:
            update[k] = new
    return update

def update_set(current_value, new_value, states):
    """Set Updater

    Returns:
        The value provided in ``new_value``.
    """
    return new_value

def update_accumulate(current_value, new_value, states):
    """Accumulate Updater

    Returns:
        The sum of ``current_value`` and ``new_value``.
    """
    return current_value + new_value

def update_nonnegative_accumulate(current_value, new_value, states):
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


#: Maps updater names to updater functions
updater_registry.register('accumulate', update_accumulate)
updater_registry.register('set', update_set)
updater_registry.register('merge', update_merge)
updater_registry.register('nonnegative_accumulate', update_nonnegative_accumulate)



## divider functions
def divide_set(state):
    """Set Divider

    Returns:
        A list ``[state, state]``. No copying is performed.
    """
    return [state, state]

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
    '''Assert that the variable is never divided

    Raises:
        AssertionError: If the variable is divided
    '''
    raise AssertionError('Variable cannot be divided')

# register dividers
divider_registry.register('set', divide_set)
divider_registry.register('split', divide_split)
divider_registry.register('split_dict', divide_split_dict)
divider_registry.register('zero', divide_zero)
divider_registry.register('zero', divide_zero)
divider_registry.register('no_divide', assert_no_divide)



# Serializers
class Serializer(object):
    def serialize(self, data):
        return data

    def deserialize(self, data):
        return data

class NumpySerializer(Serializer):
    def serialize(self, data):
        return data.tolist()

    def deserialize(self, data):
        return np.array(data)

class NumpyScalarSerializer(Serializer):
    def serialize(self, data):
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return float(data)
        raise ValueError(
            'Cannot serialize numpy scalar {} of type {}.'.format(
                data, type(data)
            )
        )

    def deserialize(self, data):
        if isinstance(data, int):
            return np.int64(data)
        if isinstance(data, float):
            return np.float64(data)
        raise ValueError(
            'Cannot deserialize scalar {} of type {}.'.format(
                data, type(data)
            )
        )

class UnitsSerializer(Serializer):
    def serialize(self, data):
        return data.magnitude

class ProcessSerializer(Serializer):
    def serialize(self, data):
        return dict(data.parameters, _name = data.name)

class GeneratorSerializer(Serializer):
    def serialize(self, data):
        return dict(data.config, _name = str(type(data)))

class FunctionSerializer(Serializer):
    def serialize(self, data):
        return str(data)

# register serializers in the serializer_registry
serializer_registry.register('numpy', NumpySerializer())
serializer_registry.register('numpy_scalar', NumpyScalarSerializer())
serializer_registry.register('units', UnitsSerializer())
serializer_registry.register('process', ProcessSerializer())
serializer_registry.register('compartment', GeneratorSerializer())
serializer_registry.register('function', FunctionSerializer())
