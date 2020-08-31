===============================
Simulation Data Representations
===============================

We represent simulation data in three ways:

* Raw Data
* Embedded Timeseries, which are sometimes called just "timeseries"
* Path Timeseries

We describe each in more detail below.

--------
Raw Data
--------

The raw data is a dictionary with times as keys and the simulation state
at that time as a dictionary. For example:

.. code-block:: python

    {
        0: {
            'agents': {
                'agent1': {
                    'boundary': {
                        'volume': 20,
                    },
                },
                'agent2': {
                    'boundary': {
                        'volume': 10,
                    },
                },
            },
        },
        1: {
            'agents': {
                'agent1': {
                    'boundary': {
                        'volume': 30,
                    },
                },
                'agent2': {
                    'boundary': {
                        'volume': 20,
                    },
                },
            },
        },
    }

You can get this data from an :term:`emitter` using its
:py:meth:`vivarium.core.emitter.Emitter.get_data` method. We recommend
keeping the main copy of your data in this form throughout your code
because it can be transformed into either of the other two forms using
the functions:

* :py:func:`vivarium.core.emitter.path_timeseries_from_data` to get a
  path timeseries
* :py:func:`vivariumcore.emitter.timeseries_from_data` to get an
  embedded timeseries

-------------------
Embedded Timeseries
-------------------

.. note:: Embedded timeseries are sometimes called just "timeseries."

An embedded timeseries is a dictionary with the same form as the
simulation state dictionary, only with an additional time key. Each
:term:`variable` in the dictionary is a key nested arbitrarily deep
within the state dictionary. Each of these keys has as its value a list
of the variable's values at each time in the list of timepoints
associated with the ``time`` key. For example:

.. code-block:: python

    {
        'agents': {
            'agent1': {
                'boundary': {
                    'volume': [20, 30],
                },
            },
            'agent2': {
                'boundary': {
                    'volume': [10, 20],
                },
            },
        },
        'time': [0, 1],
    }

You can get data in this format from an :term:`emitter` using its
:py:meth:`vivarium.core.emitter.Emitter.get_timeseries` function.

---------------
Path Timeseries
---------------

A path timeseries is a flattened form of an embedded timeseries. We take
each variable and its list of timepoints from an embedded timeseries and
make each its own entry in the dictionary. The keys are tuples
specifying the paths to each variable, and the values are the lists of
timepoints. Like in embedded timeseries, we also have a ``time`` key
with the time values for each timepoint. For example:


.. code-block:: python

    {
        ('agents', 'agent1', 'boundary', 'volume'): [20, 30],
        ('agents', 'agent2', 'boundary', 'volume'): [10, 20],
        'time': [0, 1],
    }
