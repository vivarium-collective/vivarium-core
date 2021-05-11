=======================
Projects Using Vivarium
=======================

Systems biologists have already used Vivarium to build a diverse array
of models, and our community of users has produced numerous processes
that you can use. Here we list some examples that our users have told us
about.

.. note::
   These projects may not be affiliated with the team that develops
   Vivarium Core, and the descriptions below have not been verified by
   the Vivarium Core Authors. A project's appearance here does not mean
   it has been endorsed by the Vivarium Core team, nor has the team
   verified the accuracy of the project description.

Vivarium-chemotaxis
===================

`Vivarium-chemotaxis
<https://github.com/vivarium-collective/vivarium-chemotaxis>`_ is a
library for the multi-scale model of chemotaxis described in: `Agmon,
E.; Spangler, R.K. A Multi-Scale Approach to Modeling E. coli
Chemotaxis. Entropy 2020, 22, 1101.
<https://www.mdpi.com/1099-4300/22/10/1101>`_

.. figure::
   https://raw.githubusercontent.com/vivarium-collective/vivarium-chemotaxis/master/doc/_static/ecoli_master.png

   The **Chemotaxis Master Composite**, with processes for metabolism
   (MTB), transport (TXP), transcription (TSC), translation (TRL),
   complexation (CXN), degradation (DEG), proton motive force (PMF),
   flagella activity (FLG), and chemoreceptor activity (CHE).  This
   repository includes the processes for CHE, FLG, and PMF; the other
   processes are imported from `vivarium-cell
   <https://github.com/vivarium-collective/vivarium-cell>`_.

You can use Vivarium-chemotaxis by installing it as a library:

.. code-block:: console

   $ pip install vivarium-chemotaxis

For more information, check out the `project on GitHub
<https://github.com/vivarium-collective/vivarium-chemotaxis>`_
