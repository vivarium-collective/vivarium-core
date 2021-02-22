"""
=======================
Swap Processes Process
=======================
"""

import os
from typing import Any, Dict

from vivarium.core.process import (
    Deriver,
    Composer,
)
from vivarium.core.composition import (
    simulate_composer,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium.composites.toys import ExchangeA

NAME = 'swap_compartment'


class SwapProcesses(Deriver):
    """ SwapProcesses Process

    Replaces the contents of a compartment when the state under the
    'trigger' port is set to True.

    Configuration:

    * **``removed_processes``**: A list of the names of the processes
      that will be removed from the compartment upon trigger.
    * **``new_compartment``**: An instantiated compartment, with
      processes and a topology that will replace the removed_processes
      upon trigger.
    * **``initial_state``**: states that will be set upon trigger.

    :term:`Ports`:

    * **``trigger``**: contains the variable that triggers the swap when True.
    * **``self``**: This port connects to the compartment's path, and
      this is where the `_delete` and `_generate` updates get pointed
      to upon trigger.
    """
    name = NAME
    defaults: Dict[str, Any] = {
        'removed_processes': [],
        'new_compartment': None,
        'initial_state': {},
    }

    def __init__(self, parameters=None):
        super(SwapProcesses, self).__init__(parameters)
        self.removed_processes = self.parameters['removed_processes']
        self.new_compartment = self.parameters['new_compartment']
        self.initial_state = self.parameters['initial_state']

    def ports_schema(self):
        return {
            'trigger': {
                '_default': False,
                '_emit': True},
            'self': {
                '*': {}}}

    def next_update(self, timestep, states):
        if states['trigger']:
            update = {
                'self': {
                    '_delete': self.removed_processes}}
            if self.new_compartment:
                network = self.new_compartment.generate({})  # todo -- pass in config?
                update['self'].update({
                    '_generate': [{
                        'processes': network['processes'],
                        'topology': network['topology'],
                        'initial_state': self.initial_state,
                    }]})
            return update
        else:
            return {}


# test
class ToyDeadCompartment(Composer):
    defaults = {
        'secrete': {
            'secrete_rate': 0.1}}

    def generate_processes(self, config):
        return {
            'secrete': ExchangeA(config['secrete'])}

    def generate_topology(self, config):
        return {
            'secrete': {
                'internal': ('internal',),
                'external': ('external',)}}


class ToyLivingCompartment(Composer):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'death': {
            'removed_processes': [
                'exchange',
                'death'],
            'new_compartment': ToyDeadCompartment({})
        }}

    def generate_processes(self, config):
        return {
            'exchange': ExchangeA(config['exchange']),
            'death': SwapProcesses(config['death'])}

    def generate_topology(self, config):
        self_path = ('..', config['agent_id'])
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'death': {
                # set the trigger to be the 'dead' state
                'trigger': ('dead',),
                'self': self_path}}


def test_death():
    agent_id = '1'

    # make the compartment
    compartment = ToyLivingCompartment({
        'agent_id': agent_id})

    # initial state
    initial_state = {
        'agents': {
            agent_id: {
                'external': {'A': 1},
                'trigger': False}}}

    # timeline turns death on
    time_dead = 5
    time_total = 10
    timeline = [
        (0, {('agents', agent_id, 'dead'): False}),
        (time_dead, {('agents', agent_id, 'dead'): True}),
        (time_total, {})]

    # simulate
    settings = {
        'outer_path': ('agents', agent_id),
        'timeline': {
            'timeline': timeline},
        'initial_state': initial_state}
    output = simulate_composer(
        compartment,
        settings)

    # external starts at 1, goes down until death, and then back up
    # internal does the inverse
    external_a = output['agents']['1']['external']['A']
    internal_a = output['agents']['1']['internal']['A']
    assert external_a[0] == 1
    assert external_a[time_dead] < external_a[0]
    assert external_a[time_total] > external_a[time_dead]
    assert internal_a[0] == 0
    assert internal_a[time_dead] > internal_a[0]
    assert internal_a[time_total] < internal_a[time_dead]

    return output


def run_death():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    os.makedirs(out_dir, exist_ok=True)
    output = test_death()
    plot_simulation_output(output, {}, out_dir)


if __name__ == '__main__':
    run_death()
