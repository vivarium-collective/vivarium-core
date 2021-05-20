===============================================
Welcome to the documentation for Vivarium Core!
===============================================

The Vivarium Core library provides the Vivarium interface and engine
for composing and simulating integrative, multiscale models.

.. image:: ./_static/hierarchy.png
    :width: 90%
    :align: center

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started.rst
   getting_started_dev.rst
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
framework for building biological systems in-silico.

Vivarium does not include any specific modeling frameworks, but instead
focuses on the interface between such frameworks, and provides a powerful
multiscale simulation engine that combines and runs them. Users of Vivarium
can therefore implement any type of model module they prefer -- whether it
is a custom module of a specific biophysical system, a configurable model
with its own standard format, or a wrapper for an off-the-shelf library. The
multiscale engine supports complex agents operating at multiple timescales,
and facilitates parallelization across multiple CPUs.


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
