==============================
The Vivarium Modeling Paradigm
==============================

Our goal in building Vivarium is to help you leverage the molecular
detail of whole-cell models in simulating the dynamics of populations of
cells. To achieve this, we combined ideas from agent-based modeling,
whole-cell modeling, and multiscale modeling.

An :term:`agent-based model` contains simple agents whose interactions
give rise to complex population dynamics. The behavior of the population
is an emergent property, not one that is explicitly modeled. In
Vivarium, we took inspiration from agent-based modeling in two major
ways. First, we model cell populations by modeling individual cells and
letting those cells interact in a shared environment. Second, we avoid
specifying population behavior. Instead, we want realistic population
dynamics to emerge from our modeling of the individual cells.

Unlike in agent-based modeling, Vivarium's models of individual cells
can have all the complexity of a :term:`whole-cell model`.  Whole-cell
modeling simulates phenotypes at the scale of the cell by modeling the
cell's molecular mechanisms. Whole-cell models are also a hybrid
modeling framework that integrates different mathematical
representations of mechanism. In Vivarium, we model each of these
molecular mechanisms by a :term:`process`. We can then wire processes
together to form :term:`compartments`.

Since we are modeling phenomena that occur at spatial and temporal
scales all the way from chemical reactions to cell population dynamics,
we need to run our simulations at different scales as well. To do so, we
draw from :term:`multiscale models`, which contain sub-models that
operate at different spatial and temporal scales. Some of these scales
are illustrated below:

.. image:: /_static/intro.png
   :width: 100%
   :align: center
   :alt: At the top we see a colony of bacteria. A zoomed-in view in the
       middle shows a single bacterium. Another zoomed-in view at the
       bottom shows the proteins of the bacterium in the middle.

We describe temporal scales by :term:`timesteps`, which define how
finely we discretize time.  Each process and each connection between
them have a timestep.

.. todo:: How does the bit about compartment timesteps change now?
