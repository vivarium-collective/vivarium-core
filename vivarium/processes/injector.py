"""
================
Injector Process
================

Injectors model the appearance of substances within a cell or its
environment. For example, we could inject antibiotics into a cell. More
commonly, though, injector :term:`processes` let us build toy examples.
As such, we use injectors in the documentation to keep our examples
simple.
"""


from __future__ import absolute_import, division, print_function

import os

from vivarium.core.composition import (
    simulate_process,
    TEST_OUT_DIR,
    save_timeseries,
)
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium.core.process import Process

NAME = 'injector'


class Injector(Process):

    name = NAME

    def __init__(self, initial_parameters=None):
        """Models the direct injection of substrates into a cell

        :term:`Ports`:

        * **internal**: The :term:`store` into which the substrates will
          be injected.

        .. note:: Each of these processes only supports injecting into a
            single store. To inject into multiple stores, create a
            separate instance of this injector process for each store.

        Arguments:
            initial_parameters (dict): An optional configuration
                dictionary that may include the key
                ``substrate_rate_map`` whose value must be a dictionary
                mapping substrate variable names to the rate (as a
                :py:class:`float`) at which they should be injected.
                Rates are interpreted as being in units of
                :math:`\\frac{substrateVariableUnit}{timestepUnit}`, so
                for a variable with units of molarity and a timestep in
                seconds, rates should be in :math:`\\frac{M}{s}`.
        """
        if initial_parameters is None:
            initial_parameters = {}

        self.substrate_rate_map = initial_parameters['substrate_rate_map']

        super(Injector, self).__init__(initial_parameters)

    def ports_schema(self):
        return {
            'internal': {
                substrate: {
                    '_default': 0,
                    '_emit': True,
                }
                for substrate in self.substrate_rate_map
            },
        }

    def next_update(self, timestep, states):
        return {
            'internal': {
                substrate: timestep * rate
                for substrate, rate in self.substrate_rate_map.items()
            }
        }


def run_injector():
    parameters = {
        'substrate_rate_map': {'toy': 1.0},
        }
    injector = Injector(parameters)
    settings = {
        'total_time': 10,
    }
    timeseries = simulate_process(injector, settings)
    return timeseries


def test_injector():
    timeseries = run_injector()
    # Expect [0, 1, ..., 10] because 0 at start
    expected = [i for i in range(11)]
    assert expected == timeseries['internal']['toy']


def main():
    out_dir = os.path.join(TEST_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    timeseries = run_injector()
    plot_settings = {}
    plot_simulation_output(timeseries, plot_settings, out_dir)
    save_timeseries(timeseries, out_dir)


if __name__ == '__main__':
    main()
