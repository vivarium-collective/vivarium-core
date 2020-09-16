===============================================
Welcome to the documentation for Vivarium Core!
===============================================

Vivarium is a multiscale platform for simulating cells in dynamic
environments, within which they can grow, divide, and thrive.

.. image:: ./_static/snapshots_fields.png
    :width: 100%
    :alt: A sequence of six frames showing simulated *E. coli* colony
        growth. The colony grows from a single cell in the leftmost
        frame to around 100 in the rightmost.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started.rst
   guides/index.rst
   tutorials/index.rst
   reference/index.rst

.. todo:: Maybe use :ref:`genindex`

.. todo:: Maybe use :ref:`modindex`

------------
Introduction
------------

A vivarium, literally a "place of life," is a controlled environment in
which organisms can be studied. You might have encountered examples of
vivaria like an aquarium or a terrarium. The Vivarium project provides a
framework for creating a computational vivarium which can simulate
colonies of cells in shared, dynamic environments.

The platform's framework is a novel synthesis of several modeling
methodologies:

1. Whole-cell modeling, to simulate hybrid models of cells with complex
   internal organization.
2. Agent-based modeling, to simulate the interactions of many cells in a
   shared environment.
3. Multiscale simulation, to solve problems with multiple scales of time
   and space.
4. Wirable model modules, to streamline model development by decomposing
   computational representations into modules that can be reconfigured
   and recombined.

This project, Vivarium Core, is the core engine that can simulate models
built using the Vivarium interface.

.. todo:: Use cases

Using This Documentation
========================

If you want to run Vivarium, start with our :doc:`getting started guide
<getting_started>`, which will walk you through getting Vivarium up
and running.

For step-by-step instructions, check out our :doc:`tutorials
<tutorials/index>`.

For a technical deep-dive into the important concepts in Vivarium, check
out our :doc:`topic guides <guides/index>`.

If you want to look something up like the configuration options for some
process or the definition of a word, see our :doc:`reference information
<reference/index>`.
