============
Compartments
============

To model a whole cell, we need to simulate processes running
concurrently and interacting with each other. For example, we might want
to model the synthesis of ATP by a metabolism process concurrently with
the consumption of ATP by a transport process such as a sodium pump. In
Vivarium, we model this by creating a :term:`compartment`.

--------
Overview
--------

Below we see an overview of the topics we will discuss in this document.
Processes and stores are combined to form compartments as shown in panel
B, and these compartments form a tree known as the hierarchy and shown
in panel C.

.. _fig-compartment:

.. figure:: ../_static/compartment.png
   :width: 100%
   :align: center
   :alt: A figure with 3 panels lettered A through C. In panel A, we see
       a red database symbol labeled "store" and with the text
       "variable values, units, mass, children, emitters, dividers,
       updaters" within it. Below, a yellow rectangle labeled "process"
       contains the text "variable names, parameters, mechanisms." A
       black line extending from the rectangle is labeled "port". In
       panel B, we see a blue square labeled "compartment". Inside are
       two stores and two processes, with the lower store connected to
       the ports of both processes, and the upper store connected only
       to the top process. A store outside the square labeled "boundary"
       is connected to a port of the upper process. In panel C, 4
       compartments form a tree with one compartment at the top level
       and one at the bottom level. The tree's edges are formed by black
       lines to boundary stores.

   The relationships between stores, processes (panel A), and
   compartments (panel B) in the tree (panel C).

--------------------
Processes and Stores
--------------------

A compartment models cellular functions running concurrently. We model each
of these functions as a :term:`process` in the compartment. To let these
processes interact, for example by producing and consuming a shared
resource like ATP, the processes share parts of the state, called
:term:`stores`. Each store is a collection of :term:`variables` such as
cytoplasmic ATP concentration, and in the compartment we define which
process operates on which stores using a :term:`topology`.

In our ATP example, we might assign a "cytoplasm" store to both the
metabolism and sodium pump processes. Now when we simulate the
compartment, the metabolism and sodium pump processes will be changing
the same variable, the ATP concentration in the cytoplasm store. This
means that if the rate of metabolism decreases, the cytoplasmic ATP
concentration variable will drop, so the sodium pump will export less
sodium. Thus, shared stores in composites let us simulate interacting
concurrent processes.

Process and Store Implementation
================================

Processes
---------

We write processes as classes that inherit from
:py:class:`vivarium.core.process.Process`.  To create a compartment, we
create instances of these classes to make the processes we want to
compose. If a process is configurable, we might provide a dictionary of
configuration options. For information on configuring a process, see the
process's documentation, e.g.
:py:class:`vivarium.processes.ode_expression.ODE_expression`.

We uniquely name each process in the compartment. This lets us include
instances of the same process class.

Stores
------

We represent stores with the :py:class:`vivarium.core.experiment.Store`
class; see its documentation for further details.

.. tip:: To see the data held by a store, you can use the
   :py:func:`vivarium.core.experiment.Store.get_config` function. This
   returns a dictionary representation of the store's data. To show this
   dictionary more readably, use
   :py:func:`vivarium.library.pretty.format_dict`.

----------------------------
Ports Make Processes Modular
----------------------------

We don't want process creators to worry about what kind of compartment
someone will use their processes in. Conversely, if you are creating a
compartment, you should be able to use any processes you like, even if
they weren't written with your use case in mind. Vivarium achieves this
modularity with :term:`ports`.

Each process has a list of named ports, one for each store it expects.
The process can perform all its computations in terms of these ports,
and the process also provides its update using port names.  This means
that a compartment can apply each process to any collection of stores,
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

A process specifies its port names in its constructor by calling the
superclass (:py:class:`vivarium.core.process.Process`)
constructor. For example, the
:py:class:`vivarium.processes.convenience_kinetics.ConvenienceKinetics`
class contains this line:

.. code-block:: python

    super(ConvenienceKinetics, self).__init__(ports, parameters)

The ``ports`` variable takes the form of a dictionary with port names as
keys and lists of variable names as values. For example, if ``ports``
looked like this:

.. code-block:: python

    {
        'cytoplasm': ['ATP', 'sodium'],
        'extracellular': ['sodium']
    }

then the process would be declaring that it cares about the ``ATP`` and
``sodium`` variables in the ``cytoplasm`` port and the ``sodium``
variable in the ``extracellular`` port. When the process is asked to
provide an update to the model state, it is only provided the variables
it specifies. For example, it might get a model state like this:

.. code-block:: python

    {
        'cytoplasm': {
            'ATP': 5.0,
            'sodium': 1e-2,
        },
        'extracellular': {
            'sodium': 1e-1,
        },
    }

This would happen even if the store linked to the ``cytoplasm`` port
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
sub-dictionaries have port names as keys and store names as values. For
example, the topology for the ATP example we have been considering might
look like this:

.. code-block:: python

    {
        'sodium_pump': {
            'cytoplasm': 'cell',
            'extracellularSpace': 'bloodVessel',
        },
        'metabolism': {
            'cytoplasm': 'cell',
        },
    }

-------------------
Example Compartment
-------------------

To put all this information together, let's take a look at an example
compartment that combines the glucose phosphorylation process from the
:py:doc:`process-writing tutorial <../tutorials/write_process>` with
:py:class:`vivarium.processes.injector`, which lets us "inject"
molecules into a store.

.. code-block:: python

	class InjectedGlcPhosphorylation(Compartment):

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

		def __init__(self, config):
			self.config = self.defaults
			self.config.update(config)

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
:doc:`hierarhcy guide <hierarchy>`.

------------------------
Compartment Interactions
------------------------

Even though compartments represent segregated sub-models, they still
need to interact. We model these interactions using :term:`boundary
stores` between compartments. For example, the boundary store between a
cell and its environment might track the flux of metabolites between the
cell and environment compartments.

When compartments are nested, these boundary stores also exist between
the inner and the outer compartment. Thus nested compartments form a
tree whose nodes are compartments and whose edges are boundary stores. A
node's parent is its outer compartment, while its children are the
compartments within it.

Since boundary stores can also exist between compartments who share a
parent, you may find it useful to think of compartments and their
boundary stores as a bigraph (not a bipartite graph) where the tree
denotes nesting and all the edges (including those in the tree)
represent boundary stores.
