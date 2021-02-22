# Changelog


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
