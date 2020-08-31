===========
Experiments
===========

Once you have created a model using :term:`processes` and
:term:`compartments`, you might want to run simulations of your model
and save simulation parameters so you and others can reproduce your
results. Vivarium uses :term:`experiments` to define these simulations,
which can contain arbitrarily nested :term:`compartments`. For example,
you could create an experiment with an environment compartment that
contains several cell compartments. Then you could run the experiment to
see how these cells might interact in a shared environment.

--------------------
Defining Experiments
--------------------

To create an experiment, you need only instantiate the
:py:class:`vivarium.core.experiment.Experiment` class. To help others
reproduce your experiment, create a file in the ``vivarium/experiments``
directory that defines a function that generates your experiment. For
example, here is the function from
:py:mod:`vivarium.experiments.glucose_phosphorylation` to create a toy
experiment that simulates the phosphorylation of injected glucose:

.. code-block:: python

    def glucose_phosphorylation_experiment(config=None):
        if config is None:
            config = {}
        default_config = {
            'injected_glc_phosphorylation': {},
            'emitter': {
                'type': 'timeseries',
            },
            'initial_state': {},
        }
        default_config.update(config)
        config = default_config
        compartment = InjectedGlcPhosphorylation(
            config['injected_glc_phosphorylation'])
        compartment_dict = compartment.generate()
        experiment = Experiment({
            'processes': compartment_dict['processes'],
            'topology': compartment_dict['topology'],
            'emitter': config['emitter'],
            'initial_state': config['initial_state'],
        })
        return experiment

Notice that most of the function just sets up configurations. The main
steps are:

#. Instantiate the compartment that your experiment will simulate.
#. Generate the processes and topology dictionaries that describe the
   compartment using
   :py:meth:`vivarium.core.experiment.Compartment.generate`.
#. Instantiate the experiment, passing along the processes and topology
   dictionaries. We also specify the :term:`emitter` the experiment
   should send data to and the initial state of the model. If we don't
   specify an initial state, it will be constructed based on the
   defaults we specified in the compartment's processes.

.. note:: Experiments accept only a single compartment, but you can
   still simulate many compartments! The single compartment you give to
   the experiment is the root comprtment, which will often be an
   environment. This compartment will then contain the other
   compartments you want to simulate.
