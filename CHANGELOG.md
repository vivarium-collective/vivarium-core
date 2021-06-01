# Changelog


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
