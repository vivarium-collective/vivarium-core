""" Vivarium module init
Register processes, updaters, dividers, serializers upon import
"""

# import matplotlib to help fix bug with import order
import matplotlib.pyplot as plt

# import registries
from vivarium.core.registry import (
    process_registry,
    updater_registry,
    divider_registry,
    serializer_registry,
    emitter_registry,
)

# import processes
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.tree_mass import TreeMass
from vivarium.processes.timeline import TimelineProcess
from vivarium.processes.clock import Clock
from vivarium.processes.swap_processes import SwapProcesses
from vivarium.processes.remove import Remove
from vivarium.processes.divide_condition import DivideCondition
from vivarium.processes.strip_units import StripUnits
from vivarium.processes.nonspatial_environment import (
    NonSpatialEnvironment)
from vivarium.processes.molarity_deriver import (
    MolarToCounts,
    CountsToMolar,
)
from vivarium.processes.mass_adaptor import (
    CountsToConcentration,
    MassToMolar,
    MassToCount,
)

# import updaters, dividers, serializers
from vivarium.core.registry import (
    update_accumulate, update_set, update_merge, update_null,
    update_nonnegative_accumulate, update_dictionary,
    divide_set, divide_split, divide_split_dict, divide_zero,
    assert_no_divide, divide_binomial, divide_set_value,
    NumpySerializer, NumpyScalarSerializer, UnitsSerializer,
    ProcessSerializer, ComposerSerializer, FunctionSerializer,
)

# import emitters
from vivarium.core.emitter import (
    Emitter, NullEmitter, RAMEmitter, DatabaseEmitter
)

_ = plt  # suppress PyCharm's unused-import warning


# register processes
process_registry.register(DivideCondition.name, DivideCondition)
process_registry.register(MetaDivision.name, MetaDivision)
process_registry.register(TreeMass.name, TreeMass)
process_registry.register(MolarToCounts.name, MolarToCounts)
process_registry.register(CountsToMolar.name, CountsToMolar)
process_registry.register(TimelineProcess.name, TimelineProcess)
process_registry.register(
    NonSpatialEnvironment.name, NonSpatialEnvironment)
process_registry.register(SwapProcesses.name, SwapProcesses)
process_registry.register(Remove.name, Remove)
process_registry.register(Clock.name, Clock)
process_registry.register(CountsToConcentration.name, CountsToConcentration)
process_registry.register(MassToCount.name, MassToCount)
process_registry.register(MassToMolar.name, MassToMolar)
process_registry.register(StripUnits.name, StripUnits)

# register updaters
updater_registry.register('accumulate', update_accumulate)
updater_registry.register('set', update_set)
updater_registry.register('null', update_null)
updater_registry.register('merge', update_merge)
updater_registry.register(
    'nonnegative_accumulate', update_nonnegative_accumulate)
updater_registry.register('dict_value', update_dictionary)

# register dividers
divider_registry.register('binomial', divide_binomial)
divider_registry.register('set', divide_set)
divider_registry.register('split', divide_split)
divider_registry.register('split_dict', divide_split_dict)
divider_registry.register('zero', divide_zero)
divider_registry.register('no_divide', assert_no_divide)
divider_registry.register('set_value', divide_set_value)

# register serializers
serializer_registry.register('numpy', NumpySerializer())
serializer_registry.register('numpy_scalar', NumpyScalarSerializer())
serializer_registry.register('units', UnitsSerializer())
serializer_registry.register('process', ProcessSerializer())
serializer_registry.register('composer', ComposerSerializer())
serializer_registry.register('function', FunctionSerializer())

# register emitters
emitter_registry.register('print', Emitter)
emitter_registry.register('null', NullEmitter)
emitter_registry.register('timeseries', RAMEmitter)
emitter_registry.register('ram', RAMEmitter)
emitter_registry.register('database', DatabaseEmitter)
