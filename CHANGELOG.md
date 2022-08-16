# Changelog

## v1.5.2

* (#207) Fix a bug in `Store.apply_update()` that caused failures when
  `self.value` was a list and `self.units` was set.

## v1.5.1

* (#206) Support multiple emits for the same timestep and remove
  `Engine.complete()`. Instead, callers of `Engine.run_for()` must pass
  `force_complete=True` at the end of their caller-managed simulation
  loops.

## v1.4.2

* (#203) Make `Store.soures` more comprehensive, in particular by
  including dividers and flows.

## v1.4.1

* (#202) Add the `filters` argument to `data_from_database()` to allow
  further filtering of MongoDB query results.

## v1.3.1

* (#200) Inside Engine, store the Step execution layers as lists instead
  of sets to ensure deterministic execution order.
* (#201) Restore ability to pass `initial_state` keys in `settings`
  dictionaries to `composition.py` functions.

## v1.3.0

* (#198) Introduce process commands to support more interactions with
  parallel processes. Now all `Process` methods of a parallelized
  process can be queried from the parent OS process. Users can also add
  support for custom methods of their processes.

  This change also simplifies the way `Engine` handles parallel
  processes warns users when serializers are not being found
  efficiently.
* (#192) Marks `composition.py` as deprecated and ensures that the rest
  of Vivarium Core doesn't depend on it.
* (#199) Remove some Numpy dtypes that are not available on some
  platforms from serialize.py.

## v1.2.8

* (#186) Apply function to data from database emitter in `get_history_data_db`.

## v1.2.7

* (#190) makes `global_time_precision` an `Engine` parameter rather than `Engine.run_for()` keyword -- this changes the method introduced in 1.2.5. Technically a breaking API, but only from 2 versions ago.
* (#191) don't use timestep for quiet paths. This allows Engine to skip over processes that don't meet their update condition.

## v1.2.6

* (#189) track composite updates in engine so that the composite topology can be plotted after being updated.
* (#188) RAMEmitter returns query data only if it exists.

## v1.2.5

* (#183) floating-point timestep precision.
* (#182) add pytest to setup.
* (#180) Handle nan in units serializer.

## v1.2.4

* (#179) Register the injector process.
* (#178) Remove `compose_experiment()`, which was replaced by
  `Composite.merge()` long ago.

## v1.2.3

* (#176) Fix UnitsSerializer to correctly handle lists and tuples.

## v1.2.2

* (#175) `store_schema` arg to `Engine` allows us to override schema directly.

## v1.2.1

* (#172) bug fix to keep assembly_ids when passing queries to the DB.

## v1.2.0

* (#167) Make serialization more structured and robust with serializer
  classes that know what kinds of data they can serialize and
  deserialize.

## v1.1.1

* (#169) `get_history_data_db` no longer requires a 'time' key. It
  asserts an 'assembly_id' key instead.

## v1.1.0

* (#164) Add a `_no_original_parameters` configuration to `Process`
  that, if `True`, disables the copying of parameters that
  `Process.__init__()` does by default. Disabling this copying decreases
  memory usage, but it puts the user in charge of ensuring that
  parameters are not mutated.

## v1.0.2

* (#163) Fix two bugs:

  * In `divide_condition.py`, use a divider that sets the division
    variable to `False` instead of `0`.
  * In `engine.py`, fix the function that checks that every dependency
    in the flow is also in the dictionary of steps. This function did
    not correctly handle cases where steps were nested in
    sub-dictionaries.

## v1.0.1

* (#151) Make `Engine._run_steps()` public.
* (#155) Add error messages for flows that include steps not found in
  `Engine.steps` and for dividers specified as strings not found in the
  divider registry.
* (#157) Add flows and steps to the value returned by
  `Composite.generate_store()`.

## v1.0.0

* (#131) Do not assume that a process's `initial_state` should be used
  when using the Store API.
* (#132) Improve error message when no path exists between nodes in the
  hierarchy.
* (#133) Remove `Process.local_timestep()`, which is no longer being
  used.
* (#134) Support using `**` in a ports schema to connect to an entire
  subtree of the hierarchy.
* (#143) Make more methods private.
* (#147) Fix bug in process serialization to support changing the
  parameter argument name in the constructor.

## v0.4.20

* `deep_merge_combine_lists` recursive call.

## v0.4.19

* (#130) Raise an exception when a user specifies two different dividers
  for the same variable, and when serializing a process, use its
  original parameters without any changes that may have occurred since
  the process was initialized.

## v0.4.18

* (#127) build a `store` argument's topology views in `Engine` constructor to support the store API.

## v0.4.17

* (#126) A new method, `Engine.run_for`, can be called iteratively without completing
  processes on the front. `Engine.update` keeps the same behavior as before. `Engine.complete`
  forces all processes to complete at the current global time.

## v0.4.16

* (#125) Use the `null` divider by default for processes so that upon
  division, processes are generated by the composer, not a divider.

## v0.4.15

* (#123) Extend `Composite.merge` to use steps and flows.
* (#124) Use `set` divider by default.

## v0.4.14

* (#121) Fix a bug in `Store.divide()` where daughter cell states were
  by default the mother cell state. Instead, add support for an
  `initial_state` key in the `_divide` dictionary.
* (#122) Add an `_output` flag option to `Process.ports_schema`, which marks ports as output-only.

## v0.4.13

* Improve `topology_view` caching mechanism to speed up simulations that have a lot of division.

## v0.4.12

* Allow schema overrides to override step schemas.

## v0.4.11

* Assert that when `Engine` treats a process like a step, that process
  is actually a step.
* Use `Process.is_step()` to check whether a process is a step instead
  of using `isinstance`.

## v0.4.10

* Fix `Engine` to support flows that are nested dictionaries.

## v0.4.9

* Add `profile` argument to the `Engine` constructor. When this argument is set to `True`, the simulation, including any parallel processes, will be profiled by `cProfile`. Once `Engine.end()` is called, the profile stats will be saved to `Engine.stats` for analysis.

## v0.4.8

* Remove `self._run_steps()` from `Engine.apply_update` if `view_expire`.

## v0.4.7

* `view_expire` flag for `_add` and `_delete` updates as well.

## v0.4.6

* deepcopy mother state upon division so daughter states are ensured to not reference the same objects.
* improve `_get_composite_state` to avoid recursive deepcopy.
* add `view_expire` flag in `Store.apply_update`, and have `Engine` use this to trigger `self.state.build_topology_views`.

## v0.4.5

* Fixed a major bug introduced by `topology_view` in v0.4.3, which had glob (*) schema unable to view sub-stores.
* `dict_value` updater added to `updater_registry`.

## v0.4.4

* Add queries to `Emitter`, and to `get_history_data_db`. This allows you to selectively retrieve from paths
 in the data, and can save a lot of time when retrieving data from large experiments.

## v0.4.3

* Bug fix in the new `topology_view`, return `Store` for `'*'` schema.

## v0.4.2

* Add `topology_view` caching in `Store` to improve performance.

## v0.4.1

* Fix a bug in `Store.generate()` that caused conflicts between a user-provided initial state and a schema to raise an error instead of the initial state taking priority.

## v0.4.0

* Replaces `Deriver`s with `Step`s. While derivers were executed sequentially, steps are executed in topological generations according to a dependency graph. This lets some derivers run in parallel. This change mostly preserves backwards-compatibility since `Deriver` is now an alias for `Step`, and we still support legacy derivers that are specified without dependencies. These legacy derivers are executed sequentially before any steps. However, the minor version is incremented because the following public interfaces have changed (though we don't expect this to break dependent code):

  * `Composite`s now have 2 more keys: `steps` and `flow`.
  * `Engine.run_derivers()` has been replaced by `Engine._run_steps()`.
  * `Engine.deriver_paths` has been replaced by `Engine._step_paths`.
  * New lists of step and flow updates have been added to the tuple returned by `Store.apply_update()`.
  * `Store.EMPTY_UPDATES` has two more `None` values and is now private (`Store._EMPTY_UPDATES`).
  * `Store.get_processes()` no longer returns steps (formerly "derivers"). Instead, these are returned by `get_steps()`.
  * Makes `Store.generate_paths()` private (now `Store._generate_paths()`).
  * Adds required `step` and `flow` arguments to `Store.generate()`.
  * Adds `metadata` argument to `Engine`.

* Fixes a bug where parallel derivers were re-instantiated every timestep.
* Marks the Store API as experimental, including the public use of `Store.move()`, `Store.insert()`, `Store.divide()`, and `Store.delete()`.

## v0.3.14

* De/serializer for np.bool_

## v0.3.13

* Fix topology plot to avoid using `plt.figure` so that plots work correctly when other plotting functions are used in the same Python session (e.g. Jupyter notebooks).

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
