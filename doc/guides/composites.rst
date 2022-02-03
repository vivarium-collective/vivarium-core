==========
Composites
==========

Once you start building models with Vivarium, you will probably discover
groups of processes that you want to reuse. For example, if you have a
group of processes that model a cell, you might want to create many
instances of those processes to a collection of cells. In Vivarium, we
support grouping processes using :term:`composites`.

.. note::

    The terminology here can be tricky because "process" can refer both
    to the process object and to the process class. However, for
    composites, the analogous ideas have different names: "composer"
    and "composite." You can use a composer object to generate different
    composite objects depending on what configuration you provide.
    Therefore, composers are like process classes while composites are
    like process objects.


--------------------
Processes and Stores
--------------------

A model in Vivarium consists of state variables, which are grouped into
collections called :term:`stores`, and :term:`processes` which mutate
those variables at each timestep. A :term:`topology` defines which
processes operate on which stores.

For example, consider a variable tracking ATP concentrations. We might
assign a "cytoplasm" store to two processes: one for metabolism and one
for sodium pumps.  Now when we run a simulation the metabolism and
sodium pump processes will be changing the same variable, the ATP
concentration in the cytoplasm store. This means that if the rate of
metabolism decreases, the cytoplasmic ATP concentration variable will
drop, so the sodium pump will export less sodium. Thus, shared stores in
composites let us simulate interacting concurrent processes.

Process and Store Implementation
================================

Processes
---------

We write processes as classes that inherit from
:py:class:`vivarium.core.process.Process`.  To create a composite, we
create instances of these classes to make the processes we want to
compose. If a process is configurable, we might provide a dictionary of
configuration options. For information on configuring a process, see the
process's documentation, e.g.
:py:class:`vivarium.processes.tree_mass.TreeMass`.

We uniquely name each process in the composite. This lets us include
instances of the same process class.

Stores
------

We represent stores with the :py:class:`vivarium.core.store.Store`
class; see its documentation for further details. Note that when
constructing a composite, you don't need to create the stores. Instead,
the Vivarium engine automatically creates the stores based on the
topology you specify.

.. tip:: To see the data held by a store, you can use the
   :py:func:`vivarium.core.store.Store.get_config` function. This
   returns a dictionary representation of the store's data. To show this
   dictionary more readably, use
   :py:func:`vivarium.library.pretty.format_dict`.

----------------------------
Ports Make Processes Modular
----------------------------

We don't want process creators to worry about what kind of simulation
someone will use their processes in. Conversely, if you are creating a
composite, you should be able to use any processes you like, even if
they weren't written with your use case in mind. Vivarium achieves this
modularity with :term:`ports`.

Each process has a list of named ports, one for each store it expects.
The process can perform all its computations in terms of these ports,
and the process also provides its update using port names. This means
that a composite can apply each process to any collection of stores,
making processes modular.

This modularity is analogous to the modularity of Python functions.
Think of each process as a function like this:

.. code-block:: python

    def sodium_pump(cytoplasm, extracellularSpace):
        ...
        return "Update: Decrease ATP concentration in cytoplasm by x mM"

A function's modularity comes from the fact that we can pass in different
objects for the ``cytoplasm`` parameter, even objects the function
authors hadn't thought of. ``cytoplasm`` is like the port, to which we
can provide any store we like.


----------
Topologies
----------

How do we specify which store goes with which port? To continue the
function analogy from above, we need something analogous to this:

.. code-block:: python

    cell = Cell()
    bloodVessel = BloodVessel()
    # We need something like the line below
    update = sodium_pump(cytoplasm=cell, extracellularSpace=bloodVessel)

When we call ``sodium_pump``, we specify which objects go with which
parameters. Analogously, we specify the mapping between ports and stores
using a :term:`topology`.

Defining Topologies
===================

We define topologies as dictionaries with process names as keys and
dictionaries (termed "sub-dictionaries") as values. These
sub-dictionaries have port names as keys and paths to stores as values.
For example, the topology for the ATP example we have been considering
might look like this:

.. code-block:: python

    {
        'sodium_pump': {
            'cytoplasm': ('cell',),
            'extracellularSpace': ('bloodVessel',),
        },
        'metabolism': {
            'cytoplasm': ('cell',),
        },
    }


Advanced Topologies
===================

The syntax used for declaring paths is a Unix-style tuple, with every
element in the tuple going further down the path from the root compartment,
and '..' moving up a level to an outer compartment.

.. code-block:: python

    topology = {
        'process': {
            'port1': ('path','to','store'),  # connect port1 to inner compartment
            'port2': ('..','outer_store')  # connect port2 to outer compartment
        }
    }

You can splitting a port into multiple stores. Variables read through the same
port can come from different stores. To do this, the port is mapped to a
dictionary with a ``_path`` key that specifies the path to the default store.
Variables that need to be read from different stores each get their own path in
that same dictionary. This same approach can be used to remap variable names, so
different processes can use the same variable but see it with different names.

.. code-block:: python

    topology = {
        # split a port into multiple stores
        'process1': {
            'port': {
                '_path': ('path_to','default_store'),
                'rewired_variable': ('path_to','alternate_store')
            }
        }
        # mapping variable names in process to different name in store
        'process2': {
            'port': {
                '_path': ('path to','default_store'),
                'variable_name': 'new_variable_name'
            }
        }
    }

---------------------------------
Flows for Ordered Step Operations
---------------------------------

Processes have one major drawback: you cannot specify when or in what
order they run. Processes can request timesteps, but the Vivarium engine
may not honor that request. This behavior can be problematic when you
have operations that need to run in a particular order. For example,
imagine that you want to model transcription and chromosome replication
in a bacterium. It seems natural to have a transcription process and
another replication process, but then how do you handle collisions
between the replisome and the RNA Polymerase (RNAP)? You might want to
say something like "If a replisome and RNAP collide, remove the RNAP
from the chromosome." To support this kind of statement, you can create
a :term:`step`.


Flows
=====

When constructing a composite of many :term:`steps`, you may find that some
steps depend on other steps. For example, you might have one step that
calculates the cell's mass and another step that calculates the cell's
volume based on that mass. Vivarium supports these dependencies, which
you can specify in a flow. Flows have the same structure as topologies,
but instead of their leaf values being paths, they are lists of paths
where each path specifies a dependency step. For example, this flow
would represent our mass-volume dependency:

.. code-block:: python

    {
        'mass_calculator': [],
        'volume_calculator': [('mass_calculator',)],
    }

The simulation engine will automatically figure out what order to run
the steps in such that the dependencies in the flow are respected. Note
that if two orderings both respect the flow, you should not assume that
the engine will pick one of the two orderings.

.. note::
   Step updates are applied immediately after the step executes, which
   is unlike process updates.

---------
Composers
---------

Most of the time, you won't need to create composites directly. Instead,
you'll create composers that know how to generate composites. To create
a composer, you need to define a composer class that inherits from
:py:class:`vivarium.core.composer.Composer` and implements the
:py:meth:`vivarium.core.composer.Composer.generate_processes` and
:py:meth:`vivarium.core.composer.Composer.generate_topology` methods.
``generate_processes`` should return a mapping from process names to
instantiated process objects, while ``generate_topology`` should return
a topology.

Example Composer
================

To put all this information together, let's take a look at an example
composer that combines the glucose phosphorylation process from the
:py:doc:`process-writing tutorial <../tutorials/write_process>` with an
injector, which lets us "inject" molecules into a store.

.. code-block:: python

	class InjectedGlcPhosphorylation(Composer):

		defaults = {
			'glucose_phosphorylation': {
				'k_cat': 1e-2,
			},
			'injector': {
				'substrate_rate_map': {
					'GLC': 1e-4,
					'ATP': 1e-3,
				},
			},
		}

		def generate_processes(self, config):
			injector = Injector(self.config['injector'])
			glucose_phosphorylation = GlucosePhosphorylation(
				self.config['glucose_phosphorylation'])

			return {
				'injector': injector,
				'glucose_phosphorylation': glucose_phosphorylation,
			}

		def generate_topology(self, config):
			return {
				'injector': {
					'internal': ('internal', ),
				},
				'glucose_phosphorylation': {
					'cytoplasm': ('cell', ),
					'nucleoside_phosphates': ('cell', ),
					'global': ('global', ),
				},
			}

Notice how we use the ``generate_processes`` function to create a
dictionary that maps process names to instantiated and configured
process objects. Similarly, we use ``generate_topology`` to create a
dictionary that maps port names to stores. To create steps and flows,
use the ``generate_steps`` and ``generate_flow`` methods.

You may wonder why we identify stores with tuples. In more complex
compartments, these tuples could contain many elements that specify a
kind of file path. We represent the total model state as a tree, and we
can create a store at any node to represent the sub-tree rooted at that
node. This tree is analogous to directory trees on a filesystem, and we
use tuples of store names to specify a path through this tree. We call
this tree the hierarchy, and we discuss it in more detail in the
:doc:`hierarchy guide <hierarchy>`.
