=========
Processes
=========

You should interpret words and phrases that appear fully capitalized in
this document as described in :rfc:`2119`. Here is a brief summary of
the RFC:

* "MUST" indicates absolute requirements. Vivarium may not work
  correctly if you don't follow these.
* "SHOULD" indicates strong suggestions. You might have a valid reason
  for deviating from them, but be careful that you understand the
  ramifications.
* "MAY" indicates truly optional features that you can include or
  exclude as you wish.

Models in Vivarium are built by combining :term:`processes`, each of
which models a mechanism in the cell. These processes can be combined in
a :term:`compartment` to build more complicated models. Process models are
defined in a class that inherit from
:py:class:`vivarium.core.process.Process`, and these
:term:`process classes` can be instantiated to create individual
processes.  During instantiation, the process class may accept
configuration options.

.. note:: Processes are the foundational building blocks of models in
   Vivarium, and they should be as simple to define and compose as
   possible.

---------------
Process Classes
---------------

Each process class MUST implement the API that we describe below.

Class Variables
===============

Each process class SHOULD define default configurations in a
``defaults`` class variable. The constructor SHOULD read these defaults.
For example:

.. code-block:: python

    class MyProcess:
        defaults = {
            'growth_rate': 0.0006,
        }


Constructor
===========

The constructor of a process class MUST accept as its first positional
argument an optional dictionary of configurations. If the process class
is configurable, it SHOULD accept configuration options through this
dictionary.

In the constructor, the process class MUST call its superclass
constructor with its :term:`ports` and a dictionary of parameters.

.. _constructor-define-ports:

Defining Ports
--------------

Ports MUST be specified as a dictionary with port names as keys and
lists of :term:`variable` names as values. These port names may be
chosen arbitrarily. Variable names are also at the discretion of the
process class author, but note that if two processes are to be combined
in a :term:`compartment` and share variables through a shared
:term:`store`, the processes MUST use the same variable names for the
shared variables.

.. note:: Variables always have the same name, no matter which process
    is interacting with them. This is unlike stores, which can take on
    different port names with each process.

Passing Parameters to Superclass Constructor
--------------------------------------------

The dictionary of parameters SHOULD include any configuration options
not used by the process class. Any information needed by the process
class MAY also be included in these parameters. Once the object has
been instantiated, these parameters are available as
``self.parameters``, where they have been stored by the
:py:class:`vivarium.core.process.Process` constructor.

Example Constructor
-------------------

Let's examine an example constructor from a growth process class.

.. code-block:: python

    def __init__(self, initial_parameters={}):
        ports = {
            'global': ['mass', 'volume']}

        parameters = {'growth_rate': self.defaults['growth_rate']}
        parameters.update(initial_parameters)
        super(Growth, self).__init__(ports, parameters)

In this constructor, only one port, ``global``, is defined, from which
the process will only need the ``mass`` and ``volume`` variables. While
the default growth rate is ``0.0006``, this can be overridden by
including a ``growth_rate`` key in the configuration dictionary passed
to ``initial_parameters``.

.. note:: ``global`` is a special port used by :term:`derivers`. It
    stores information about the total model state that, like ``mass``
    doesn't fit into any store.

.. _constructor-ports-schema:

Ports Schema
============

The process class MUST implement a ``process_schema`` method with no
required arguments. This method MUST return nested dictionaries of the
following form:

.. code-block:: python

    {
        'port_name': {
            'variable_name': {
                'schema_key': 'schema_value',
                ...
            },
            ...
        },
        ...
    }

``schema_key`` MUST be a :term:`schema key` and have an appropriate
value. Any applicable and omitted schema keys will take on their default
values. Note that every variable SHOULD specify ``_default``. If the
cell will be dividing, every variable also MUST specify ``_divider``.
Variables in the ports schema SHOULD NOT specify ``_value``.

Example Ports Schema
--------------------

.. code-block:: python

    def ports_schema(self):
        return {
            'global': {
                'mass': {
                    '_emit': True,
                    '_default': 1339 * units.fg,
                    '_updater': 'set',
                    '_divider': 'split'},
                'volume': {
                    '_updater': 'set',
                    '_divider': 'split'},
                'divide': {
                    '_default': False,
                    '_updater': 'set'
                }
            }
        }

Here we specify that only ``mass`` should be emitted. We assign a
default value of 1339 fg to ``mass``, and we declare that the ``mass``
and ``volume`` variables should be split in half on division. Further,
we specify that all the three variables should have their updates set,
not accumulated.

Derivers
========

For each port, we can also specify a :term:`deriver`. Each process class
MUST implement a `derivers` method that returns a dictionary whose keys
are the ports to which we want to apply derivers. For each port, the
value in the dictionary must be a dictionary with the following keys:

* **deriver** (:py:class:`str`): The name of the deriver to apply.
* **port_mapping** (:py:class:`dict`): Maps from ports of the deriver
  process to ports of the process class we are writing. This is like a
  :term:`topology`.
* **config** (:py:class:`dict`): A configuration dictionary that
  conforms to the requirements of the particular deriver being invoked.

Next Updates
============

Each process class MUST implement a ``next_update`` method that accepts
two positional arguments: the :term:`timestep` and the current state of
the model. The timestep describes, in units of seconds, the length of
time for which the update should be computed.

State Format
------------

The ``next_update`` method MUST accept the model state as a dictionary
of the same form as the :ref:`default state dictionary
<constructor-ports-schema>`, but with the dictionary of schema keys
replaced with the current (i.e. pre-update) value of the variable.

.. note:: In the code, you may see the model state referred to as
    ``states``. This is left over from when stores were called states,
    and so the model state was a collection of these states. As you may
    already notice, this naming was confusing, which is why we now use
    the name "stores."

Because of :term:`masking`, each
port will contain only the variables specified in the
:ref:`constructor's ports declaration <constructor-define-ports>`, even
if the linked store contains more variables.

.. WARNING:: The ``next_update`` method MUST NOT modify the states it is
    passed in any way. The state's variables are not copied before they
    are passed to ``next_update``, so changes to any objects in the
    state will affect the model state before the update is applied.

Update Format
-------------

``next_update`` MUST return a single dictionary, the update that
describes how the modeled mechanism would change the model state over
the specified time. The update dictionary MUST be of the same form as the
:ref:`default state dictionary <constructor-ports-schema>`, though with
the dictionaries of schema keys replaced with update values. Also,
variables that do not need to be updated can be excluded.

Example Next Update Method
--------------------------

Here is an example ``next_update`` method for our growth process:

.. code-block:: python

    def next_update(self, timestep, states):
        mass = states['global']['mass']
        new_mass = mass * np.exp(self.parameters['growth_rate'] * timestep)
        return {'global': {'mass': new_mass}}

Recall from :ref:`our example schema <constructor-ports-schema>` that we use
the ``set`` updater for the ``mass`` variable. Thus, we compute the new
mass of the cell and include it in our update. Notice that we access the
growth rate specified in the constructor by using the
``self.parameters`` attribute.

.. note:: Notice that this function works regardless of what timestep we
    use. This is important because different compartments may need
    different timesteps based on what they are modeling.

Process Class Examples
======================

Many of our process classes have examples in the form of test functions
at the bottom. These are great resources if you are trying to figure out
how to use a process.

If you are writing your own process, please include these examples!
Also, executing the process class Python file should execute one of
these examples and save the output as demonstrated in
:py:mod:`vivarium.processes.convenience_kinetics`. Lastly, any top-level
functions you include that are prefixed with ``test_`` will be executed
by ``pytest``. Please add these tests to help future developers make
sure they haven't broken your process!

---------------------
Using Process Objects
---------------------

Your use of process objects will likely be limited to instantiating them
and passing them to other functions in Vivarium that handle running the
simulation. Still, you may find that in some instances, using process
objects directly is helpful. For example, for simple processes, the
clearest way to write a test may be to run your own simulation loop.

Simulating a process can be sketched by the following pseudocode:

.. code-block:: python

    # Create the process
    configuration = {...}
    process = ProcessClass(configuration)

    # Get the initial state from the process's schema
    # This means the stores and ports are the same
    state = {}
    schema = process.ports_schema()
    for port, port_dict in schema.items():
        for variable, variable_schema in port_dict.items():
            state[port][variable] = variable_schema["_default"]

    # Run the simulation in a loop for 10 seconds
    time = 0
    while time < 10:
        # We are using a timestep of 1 second
        update = process.next_update(1, state)
        # This is a simplified way to apply the update that assumes all
        # all variables are numbers and all updaters are "accumulate"
        for port in update:
            for variable_name, value in port.items():
                state[port][variable_name] += value
    # Now that the loop is finished, the predicted state after 10
    # seconds is in "state"

The above pseudocode is simplified, and for all but the most simple
processes you will be better off using Vivarium's built-in simulation
capabilities. We hope though that this helps you understand how
processes are simulated and the purpose of the API we defined.
