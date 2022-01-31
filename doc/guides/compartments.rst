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

    You might be wondering why we implemented processes and composites
    differently. The reason is that for the Vivarium Engine to model
    reproduction (e.g. cell division), it needs to have a composer
    object that can generate the new composites.

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

How Processes Define Ports
==========================

A process specifies its port names in its
:py:meth:`vivarium.core.process.Process.ports_schema` method.  For
example, the :py:class:`vivarium.processes.tree_mass.TreeMass` schema is
created like this:

.. code-block:: python

    def ports_schema(self):
        return {
            'global': {
                'initial_mass': {
                    '_default': self.parameters['initial_mass'],
                    '_updater': 'set',
                    '_divider': 'split',
                },
                'mass': {
                    '_default': self.parameters['initial_mass'],
                    '_emit': True,
                    '_updater': 'set',
                    '_divider': 'split',
                },
            },
        }

The top level keys are the port names. In this case, the only port is
``global``. The next level of keys define the variables expected to be
in each port. Here, we expect ``global`` to have the variables
``initial_mass`` and ``mass``.

When the process is asked to provide an update to the model state, it is
only provided the variables it specifies. For example, it might get a
model state like this:

.. code-block:: python

    {
        'global': {
            'initial_mass': 0,
            'mass': 1339,
        },
    }

This would happen even if the store linked to the ``global`` port
contained more variables. We call this stripping-out of variables the
process doesn't need :term:`masking`.

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
dictionary that maps port names to stores.

You may wonder why we identify stores with tuples. In more complex
compartments, these tuples could contain many elements that specify a
kind of file path. We represent the total model state as a tree, and we
can create a store at any node to represent the sub-tree rooted at that
node. This tree is analogous to directory trees on a filesystem, and we
use tuples of store names to specify a path through this tree. We call
this tree the hierarchy, and we discuss it in more detail in the
:doc:`hierarchy guide <hierarchy>`.
