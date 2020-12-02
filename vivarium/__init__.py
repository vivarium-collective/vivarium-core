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
)

# import processes
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.tree_mass import TreeMass
from vivarium.processes.derive_concentrations import DeriveConcentrations
from vivarium.processes.derive_counts import DeriveCounts
from vivarium.processes.timeline import TimelineProcess
from vivarium.processes.nonspatial_environment import NonSpatialEnvironment
from vivarium.processes.swap_processes import SwapProcesses
from vivarium.processes.disintegrate import Disintegrate
from vivarium.processes.divide_condition import DivideCondition

# import updaters, dividers, serializers
from vivarium.core.registry import (
    update_accumulate, update_set, update_merge, update_nonnegative_accumulate,
    divide_set, divide_split, divide_split_dict, divide_zero, assert_no_divide,
    divide_binomial,
    NumpySerializer, NumpyScalarSerializer, UnitsSerializer, ProcessSerializer,
    GeneratorSerializer, FunctionSerializer
)


# register processes
process_registry.register(DivideCondition.name, DivideCondition)
process_registry.register(MetaDivision.name, MetaDivision)
process_registry.register(TreeMass.name, TreeMass)
process_registry.register(DeriveConcentrations.name, DeriveConcentrations)
process_registry.register(DeriveCounts.name, DeriveCounts)
process_registry.register(TimelineProcess.name, TimelineProcess)
process_registry.register(NonSpatialEnvironment.name, NonSpatialEnvironment)
process_registry.register(SwapProcesses.name, SwapProcesses)
process_registry.register(Disintegrate.name, Disintegrate)

# register updaters
updater_registry.register('accumulate', update_accumulate)
updater_registry.register('set', update_set)
updater_registry.register('merge', update_merge)
updater_registry.register('nonnegative_accumulate', update_nonnegative_accumulate)

# register dividers
divider_registry.register('binomial', divide_binomial)
divider_registry.register('set', divide_set)
divider_registry.register('split', divide_split)
divider_registry.register('split_dict', divide_split_dict)
divider_registry.register('zero', divide_zero)
divider_registry.register('no_divide', assert_no_divide)

# register serializers
serializer_registry.register('numpy', NumpySerializer())
serializer_registry.register('numpy_scalar', NumpyScalarSerializer())
serializer_registry.register('units', UnitsSerializer())
serializer_registry.register('process', ProcessSerializer())
serializer_registry.register('compartment', GeneratorSerializer())
serializer_registry.register('function', FunctionSerializer())
