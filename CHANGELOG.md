# Changelog


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
