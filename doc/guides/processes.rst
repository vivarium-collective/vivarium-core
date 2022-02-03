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
which models a mechanism in the system being studied. These processes
can be combined in a :term:`composite` to build more complicated
models. Process models are defined in classes that inherit from
:py:class:`vivarium.core.process.Process`, and these :term:`process
classes` can be instantiated to create individual processes.  During
instantiation, the process class may accept configuration options.

.. note:: Processes are the foundational building blocks of models in
   Vivarium, and they should be as simple to define and compose as
   possible.

--------------------------
Process Interface Protocol
--------------------------

Each process class MUST implement the application programming interface
(API)that we describe below.

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
constructor with a dictionary of parameters.

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

    def __init__(self, initial_parameters=None):
        if initial_parameters == None:
            initial_parameters = {}
        parameters = {'growth_rate': self.defaults['growth_rate']}
        parameters.update(initial_parameters)
        super().__init__(parameters)

Note that Vivarium Core actually handles combining the provided
parameters with the default parameters, so a constructor as simple as
the one above can actually be dropped. The superclass constructor makes
it redundant, but we show it here for clarity.

.. WARNING:: Python creates only one instance of both class variables
   and function argument defaults. This means that you MUST not change
   the default parameters object. Make a copy instead. This also means
   that you SHOULD avoid using a mutable object as a default argument.
   This is why we use ``None`` as the default for ``initial_parameters``
   instead of ``{}``.

While the default growth rate is ``0.0006``, this can be overridden by
including a ``growth_rate`` key in the configuration dictionary passed
to ``initial_parameters``.

These special parameters get handled by the superclass constructor:

* ``name``: The value of the ``name`` parameter gets assigned to the
  process's ``name`` attribute (e.g. ``my_process.name``). If no name is
  specified in the parameters or as a class variable, we use
  ``self.__class__.__name__`` as the name.
* ``time_step``: If not specified, the ``time_step`` parameter is set to
  1. This parameter determines how frequently the simulation engine runs
  this process's ``next_update`` function.
* ``_condition``: The value of this parameter should be a path in the
  ``states`` dictionary passed to ``next_update()`` to a variable. The
  variable should hold a boolean specifying whether the process's
  ``next_update`` function should run.

.. _constructor-ports-schema:

Ports Schema
============

Each process declares what stores it expects by specifying a
:term:`port` for each store it accepts. Note that if two processes are
to be combined in a model and share variables through a shared
:term:`store`, the processes MUST use the same variable names for the
shared variables.

The process class MUST implement a ``ports_schema`` method with no
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

Schema keys
-----------

``schema_key`` MUST be a :term:`schema key` and have an appropriate
value. Any applicable and omitted schema keys will take on their default
values. Note that every variable SHOULD specify ``_default``. If the
cell will be dividing, every variable also MUST specify ``_divider``.
Variables in the ports schema SHOULD NOT specify ``_value``.

Available schema keys include:

* ``_default``: The default value of the state variable if no initial value
  is provided. This also sets the data type of the variable, including units.
* ``_updater``: How to apply state variable updates. Available updaters are
  listed in below
* ``_divider``: How to divide the state variable's values between daughter
  cells. Available dividers are listed below.
* ``_emit``: A Boolean value that sets whether to log this variable to the
  simulation database for later analysis.
* ``_properties``: User-defined properties such as molecular weight. These
  can be used for calculating variables such as total system mass.

Updaters
--------

Updaters are methods by which an update from a process is applied to a variable's value.

Updaters provided by vivarium-core include:

* ``accumulate`` & The default updater. Add the update value to the current value.
* ``set`` & The update value becomes the new current value.
* ``merge`` & Update an existing dictionary with new values, and add any newly declared keys.
* ``null`` & Do not apply the update.
* ``nonnegative_accumulate`` & Add the update value to the current value, and set
  to `0` if the result is negative.
* ``dict_value`` & translates \_add and \_delete -style updates to operate on a dictionary.

New updaters can be easily defined and passed into a port schema:

.. code-block:: python

    # updater that returns a random value
    def random_updater(current_value, update_value):
        return random.random()

    def port_schema(self):
        ports = {
            'port1': {
                'variable1': {
                    '_default': 1.0
                    '_updater': {
                        'updater': random_updater
                        }
                }
            }
        }
        return ports

Dividers
--------

Dividers are methods by which a variable's value is divided when division is triggered.

Dividers available in vivarium-core include:

* ``set``: The default divider. Daughters get the same value as the mother.
* ``binomial``: Sample the first daughter's value from a binomial distribution of
  the mother's value, and the second daughter gets the remainder.
* ``split``: Divide the mother's value in two. Odd integers will make one daughter
  receive `1` more than the other daughter.
* ``split_dict``: Splits a dictionary of {key: value} pairs, with each daughter
  receiving a dictionary with the same keys, but with each value split.
* ``zero``: Daughter values are both set to `0`.
* ``no_divide``: Asserts that this value should not be divided.

New dividers can be easily defined and passed into a port schema:

.. code-block:: python

    # divider that returns a random value for each daughter
    def random_divider(mother_value, state):
        return [
            random.random(),
            random.random()]

    def port_schema(self):
        ports = {
            'port1': {
                'variable1': {
                    '_default': 1.0
                    '_divider': {
                        'divider': random_divider
                        }
                }
            }
        }
        return ports


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

Views
-----

When the process is asked to provide an update to the model state, it is
only provided the variables it specifies. For example, it might get a
model state like this:

.. code-block:: python

    {
        'global': {
            'mass': 1339 <Unit('femtogram')>,
            'volume': 1.2,
            'divide': False,
        },
    }

This would happen even if the store linked to the ``global`` port
contained more variables. We call this stripping-out of variables the
process doesn't need :term:`masking`.

Advanced Ports Schema
=====================

Use the glob ``*`` schema to declare expected sub-store structure,
and view all child values of the store:

.. code-block:: python

    schema = {
        'port1': {
            '*': {
                '_default': 1.0
            }
        }
    }

Use the glob ``**`` schema to connect to an entire sub-branch, including
child nodes, grandchild nodes, etc:

.. code-block:: python

    schema = {
        'port1': '**'
    }

Ports flagged as output-only won't be viewed through the next_update's
states, which can save some overhead time:

.. code-block:: python

    schema = {
        'port1': {
            '_output': True,
            'A': {'_default': 1.0},
        }
    }


Next Updates
============

Each process class MUST implement a ``next_update`` method that accepts
two positional arguments: the :term:`timestep` and the current state of
the model. The timestep describes, in units of seconds, the length of
time for which the update should be computed.

State Format
------------

The ``next_update`` method MUST accept the simulation state as a
dictionary of the same form as the :ref:`ports schema dictionary
<constructor-ports-schema>`, but with the dictionary of schema keys
replaced with the current (i.e. pre-update) value of the variable.

.. note:: In the code, you may see the simulation state referred to as
   ``states``. This is left over from when stores were called states,
   and so the simulation state was a collection of these states. As you
   may already notice, this naming was confusing, which is why we now
   use the name "stores."

Because of :term:`masking`, each port will contain only the variables
specified in the :ref:`ports schema <constructor-ports-schema>`, even if
the linked store contains more variables.

.. WARNING:: The ``next_update`` method MUST NOT modify the states it is
   passed in any way. The state's variables are not copied before they
   are passed to ``next_update``, so changes to any objects in the state
   will affect the simulation state before the update is applied.

Update Format
-------------

``next_update`` MUST return a single dictionary, the update that
describes how the modeled mechanism would change the simulation state
over the specified time. The update dictionary MUST be of the same form
as the :ref:`ports schema dictionary <constructor-ports-schema>`, though
with the dictionaries of schema keys replaced with update values. Also,
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
    use. This is important because different simulations may need
    different timesteps based on what they are modeling.

----------------------
Process Class Examples
----------------------

Many of our process classes have examples in the form of test functions
at the bottom. These are great resources if you are trying to figure out
how to use a process.

If you are writing your own process, please include these examples!
Also, executing the process class Python file should execute one of
these examples and save the output as demonstrated in
``vivarium.processes.glucose_phosphorylation``. Lastly, any
top-level functions you include that are prefixed with ``test_`` will be
executed by ``pytest``. Please add these tests to help future developers
make sure they haven't broken your process!

-----
Steps
-----

:term:`Step` is subclass of :term:`Process` that is not time-dependent.
These instances run before the first timestep, and after the dynamic processes
during simulation. The run according to a dependency graph called a :term:`flow`
(like a workflow) -- see :ref:`flows topic guide <constructor-flows>`.
These can serve many different roles, including translating
states between different modeling formats, implementing lift or restriction
operators to translate states between scales, and as auxiliary processes that
offload complexity. As an example of offloading complexity, a step might
recalculate concentrations after counts have been updated.

To create a step, you follow the same steps as you would to create a
:term:`process` except that your class should inherit from
:py:class:`vivarium.core.process.Step`. For example, we could create a
replisome-RNAP collision reconciler like this:

.. code-block:: python

    class CollisionReconciler(Step):

        def ports_schema(self):
            return {
                'replisomes': {
                    '*': {
                        'position': {'_default': 0},
                    },
                },
                'RNAPs': {
                    '*': {
                        'position': {'_default': 0},
                    },
                },
            }

        def next_update(self, timestep, states):
            # We can ignore the timestep since it will always be 0.
            replisome_positions
                replisome['position']
                for replisome in states['replisomes'].values()
            ])
            rnap_positions = np.array([
                rnap['position']
                for rnap in states['RNAPs'].values()
            ])
            # Assume that our timestep is small enough that we can
            # ignore RNAPs and replisomes that move past each other
            # (instead of to the same position) in one timestep.
            collision_mask = replisome_positions == rnap_positions
            rnap_keys = np.array(list(states['RNAPs'].keys()))
            to_remove = rnap_keys[collision_mask]
            return {
                'RNAPs': {
                    '_delete': to_remove.tolist(),
                },
            }

.. note::
   Steps are always given a timestep of 0 by the simulation engine.

Step Implementation Details
===========================

Steps are technically identified by whether their
:py:meth:`vivarium.core.process.Process.is_step()` methods return
``True``. This means that you can make a process that determines whether
it should be a Step based on its configuration. Note however that we do
not support changing whether a process is a step mid-simulation.

-----------------
Advanced Features
-----------------

Adaptive Timesteps
==================

You can set process timesteps for the duration of a simulation using the
``time_step`` parameter, but you can also override the
:py:meth:`vivarium.core.process.Process.calculate_timestep` method to
compute timesteps dynamically based on the same view into the simulation
state that ``next_update()`` sees.

Conditional Updates
===================

Sometimes you might want the simulation engine to skip a process when
generating updates. You can implement this by overriding
:py:meth:`vivarium.core.process.Process.update_condition` to return
``False`` whenever you don't want the process to run. This method takes
as a parameter the same view into the simulation state that
``next_update()`` sees.

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
