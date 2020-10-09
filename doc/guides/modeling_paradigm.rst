==============================
The Vivarium Modeling Paradigm
==============================

Vivarium is a novel simulation engine that addresses computational
biology's dual challenges of model reuse and multi-scale integration
by explicitly separating the interface that connects models from the
frameworks that implement them. Our goal was to make it as simple as
possible to write modular models of any dynamical system in computational
biology, run these on their own, and freely compose them with other models
in executable multi-scale representations. To achieve this, we combined
methods from agent-based modeling, whole-cell modeling, multi-scale
simulation, and modular programming.

An :term:`agent-based model` contains simple agents whose interactions
give rise to complex collective dynamics. The behavior of the population
is an emergent property, not one that is explicitly modeled. In
Vivarium, we took inspiration from agent-based modeling in two major
ways. First, models can be instantiated within computational objects, and
those objects can interact in a shared environment within which
they are embedded. Second, we avoid specifying population behavior.
Instead, we want realistic population dynamics to emerge from the
interactions of the individual sub-models.

Unlike in agent-based modeling, Vivarium's models can have all the
complexity of a :term:`whole-cell model`.  Whole-cell models utilize
a hybrid modeling approach that integrates different mathematical
representations of mechanism. In Vivarium, we model each of these
molecular mechanisms by a :term:`process`. We can then wire processes
together to form :term:`composites`. As an extension to the whole-
cell modeling framework, these can be embedded in a nested :term:`hierarchy`
of :term:`compartments`, with many distributed processes operating
on a variety of different states.

Since we are modeling phenomena that occur at spatial and temporal
scales all the way from chemical reactions to cellular populations,
we need to run our simulations at different scales as well. To do so, we
draw from :term:`multiscale models`, which simulate sub-models that
operate at different spatial and temporal scales. Some of these scales
are illustrated below:

Modular programming is a common approach in software engineering,
with computing interfaces that define the types of interactions between
independent software applications. By separating the :term:`process`
interface from the modeling framework, Vivarium allows users to develop any
type of dynamical model as a modular process and wire it together with other
processes in a compartment hierarchy. A separate simulation engine (i.e.
vivarium-core) can take the processes and execute them in time. Finally,
Vivarium embraces the modularity of software libraries so that models can be
developed for separate projects and then imported, reconfigured, and recombined
to support incremental changes that iterate on model design and build upon
previous work.

.. image:: /_static/intro.png
   :width: 100%
   :align: center
   :alt: At the top we see a colony of bacteria. A zoomed-in view in the
       middle shows a single bacterium. Another zoomed-in view at the
       bottom shows the proteins of the bacterium in the middle.