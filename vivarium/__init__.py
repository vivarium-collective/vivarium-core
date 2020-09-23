from vivarium.core.registry import process_registry

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
