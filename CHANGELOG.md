# Changelog


## v0.3.12

* add composite specification

## v0.3.11

* fix bug in serializing list

## v0.3.10

* add `emitter_registry`, so that emitters can be added from different repositories by importing the registry and registering their emitter.
* add `_condition` key to process parameters, that lets them by conditionally triggered by connecting to a boolean states in the hierarchy.

## v0.3.9

* reduce mongo document limit for the DatabaseEmitter

## v0.3.8

* revert rename of divider argument to "state"
* do not allow _add updates with repeat keys

## v0.3.7

* fix custom dividers to use topology argument

## v0.3.5

* remove unneeded deepcopy from update step, leading to runtime boost.
* runtime_profile experiment added.

## v0.3.4

* Engine uses keyword arguments rather than a config dict. To fix this do `Engine(**config_dict)` instead of `Engine(config_dict)`. Additional emit_config options are added to selectively emit metadata.

## v0.3.3

* Engine has emit_config option. Is set to False, this keeps the serialized processes from being emitted and can save a lot of compute time.

## v0.3.2

* DatabaseEmitter supports arbitrary emit sizes by breaking up large emits into chunks, and reassembles them seamlessly upon retrieval.
* libary.wrappers make_logging_process adds a logging port to any process
* Improve the Store API by adding methods for creating new stores and connecting processes through their ports. 

## v0.3.1

* Fix `plot_topology` to use `Process.generate()` now that `Process` no longer inherits from `Composer`

## v0.3.0

* Store API supports simplified scripting of a bigraph
* Breaking import changes: `Composite` and `Composer` are now under `vivarium.core.composer`, and `Experiment` has been renamed `Engine` under `vivarium.core.engine`.

## v0.2.16

* add more flexible kwarg passing in `Control` 

## v0.2.15

* make `Experiment`'s `invoke_process` method public again

## v0.2.14

* fix bug to clear Composite of processes and topology instance variables
* more general divider declaration
* add time_display arg to agents_multigen
* update docs

## v0.2.13

* handle large emit_configuration to database emitter by checking the size of the packet, and removing process parameters if necessary.

## v0.2.12

* plotting tweaks. 
* provide information about update failure.
* improve data_from_database to return experiment config.

## v0.2.11

* process.schema = None in base class constructor

## v0.2.10

* string-based path declaration for topologies.
* custom node border widths in plot_topology.

## v0.2.9

* composite.merge() uses the provided topology to override merged composite topology

## v0.2.8

* improved plot_topology

## v0.2.7

* plot_topology includes new the `graph_options`: `horizontal` and `hierarchy`, and new  `process_colors` and `store_colors` for coloring individual nodes.

## v0.2.6

* new Composite and AggregateComposer classes in core/process.py

## v0.2.5

* remove future annotations to continue support of python 3.6

## v0.2.3

* bring back compatibility for python 3.6

## v0.2.2

* generalize mass_adaptors and molarity_derivers

## v0.2.1

* add mass_adaptor

## v0.2.0

* breaking API changes:
    * Composer class instead of Composite. 
    * several composition.py functions have been renamed to improve self-description.

## v0.1.12

* bug fix when handling complex_topology. store.apply_update() can now go up paths with '..'.
* calculate_timestep() for Process supports adaptive timesteps.

## v0.1.11

* clock process initial state

## v0.1.10

* add plot_variables plotting function to simulation_output

## v0.1.9

* clock process for keeping track of global time

## v0.1.7

* growth_rate process for generic exponential growth of variables

## v0.1.6

* topology grapher option to leave edge colors black

## v0.1.5

* topology grapher uses flowchart symbols

## v0.1.4

* add options to agents_multigen plot, for improved look on Jupyter notebooks

## v0.1.3

* fix Composite generate() to get correct path

## v0.1.2

* rework the class structure of Processes and Composites. Replace Generator with Factory.

## v0.1.1

* get_networkx_graph() and graph_figure() in vivarium/plot/topology.py

## v0.0.39

* improved units handling, with a units serializer for emit_data, and convert data to timeseries stripping the units.

## v0.0.36

* multi_update for experiment allos multiple updates to a variable from the same process
* fix some inspection errors & warnings
* merge for generator class

## v0.0.34

* Control object in vivarium/cor/control for streamlining experiments
* progress bar and display in experiment
* disintegrate process
* divide_condition process
* swap_compartment process

## v0.0.29

* timeline process configured with paths for ports

## v0.0.28

* vivarium/plots directory
* register in module init
* get_initialize_state for Generator
* generalize timeline process

## v0.0.20

* Update documentation with:
  * A template repository for getting started.
  * Changes to reflect recent updates to the project.
  * Tests for the documentation.
* Add injector process.
* Add demo process, compartment, and experiment modeling glucose
  phosphorylation.
