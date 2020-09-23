"""
===============================================
Registry of Updaters, Dividers, and Serializers
===============================================

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

--------
Updaters
--------

Each :term:`updater` is defined as a function whose name begins with
``update_``. Vivarium uses these functions to apply :term:`updates` to
:term:`variables`. Updater names are registered in
:py:data:`updater_registry`, which maps these names to updater functions.

Updater API
===========

An updater function MUST have a name that begins with ``update_``. The
function MUST accept exactly three positional arguments: the first MUST
be the current value of the variable (i.e. before applying the update),
the second MUST be the value associated with the variable in the update,
and the third MUST be either a dictionary of states from the simulation
hierarchy or None if no ``port_mapping`` key was specified in the
updater definition. The function SHOULD not accept any other parameters.
The function MUST return the updated value of the variable only.

--------
Dividers
--------

Each :term:`divider` is defined by a function that follows the API we
describe below. Vivarium uses these dividers to generate daughter cell
states from the mother cell's state. Divider names are registered in
:py:data:`divider_registry`, which maps these names to divider functions.

Divider API
===========

Each divider function MUST have a name prefixed with ``_divide``. The
function MUST accept a single positional argument, the value of the
variable in the mother cell. It SHOULD accept no other arguments. The
function MUST return a :py:class:`list` with two elements: the values of
the variables in each of the daughter cells.

.. note:: Dividers MAY not be deterministic and MAY not be symmetric.
    For example, a divider splitting an odd, integer-valued value may
    randomly decide which daughter cell receives the remainder.
"""


from __future__ import absolute_import, division, print_function

import random

import numpy as np

from vivarium.library.dict_utils import deep_merge
from vivarium.library.units import Quantity, units


class Registry(object):
    def __init__(self):
        self.registry = {}

    def register(self, key, item):
        if key in self.registry:
            if item != self.registry[key]:
                raise Exception('registry already contains an entry for {}: {} --> {}'.format(key, self.registry[key], item))
        else:
            self.registry[key] = item

    def access(self, key):
        return self.registry.get(key)


#: Maps process names to :term:`process classes`
process_registry = Registry()

#: Map updater names to updater functions
updater_registry = Registry()

#: Map divider names to divider functions
divider_registry = Registry()

#: Map serializer_registry names to divider serializer_registry
serializer_registry = Registry()
