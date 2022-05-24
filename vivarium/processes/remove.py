"""
====================
Remove Process
====================
"""

import os
from typing import Any, Dict

from vivarium.core.engine import pp
from vivarium.core.process import (
    Deriver,
)
from vivarium.core.composer import Composer
from vivarium.core.directories import PROCESS_OUT_DIR
from vivarium.core.engine import Engine
from vivarium.composites.toys import ExchangeA
from vivarium.processes.timeline import TimelineProcess

NAME = 'remove'


class Remove(Deriver):
    """ Remove Process

    remove a compartment when the state under the 'trigger' port is set to True.
    """
    name = NAME
    defaults: Dict[str, Any] = {}

    def __init__(self, parameters=None):
        super(Remove, self).__init__(parameters)
        self.agent_id = self.parameters['agent_id']

    def ports_schema(self):
        return {
            'trigger': {
                '_default': False,
                '_emit': True},
            'agents': {'*': {}}}

    def next_update(self, timestep, states):
        if states['trigger']:
            return {
                'agents': {
                    '_delete': [self.agent_id]}}
        else:
            return {}


# test
class ToyLivingCompartment(Composer):
    defaults = {
        'exchange': {'uptake_rate': 0.1},
        'death': {}
    }

    def generate_processes(self, config):
        death_config = config['death']
        death_config['agent_id'] = config['agent_id']
        return {
            'exchange': ExchangeA(config['exchange']),
            'death': Remove(death_config)}

    def generate_topology(self, config):
        agents_path = ('..', '..', 'agents')
        return {
            'exchange': {
                'internal': ('internal',),
                'external': ('external',)},
            'death': {
                # set the trigger to be the 'dead' state
                'trigger': ('dead',),
                'agents': agents_path}}


def test_remove():
    agent_id = '1'

    # timeline turns death on
    time_dead = 5
    time_total = 10
    timeline = [
        (0, {('agents', agent_id, 'dead'): False}),
        (time_dead, {('agents', agent_id, 'dead'): True}),
        (time_total, {})]
    timeline_process = TimelineProcess({'timeline': timeline})

    # make the composite
    compartment = ToyLivingCompartment({
        'agent_id': agent_id})
    composite = compartment.generate(path=('agents', agent_id))
    composite.merge(
        processes={'timeline': timeline_process},
        topology={'timeline': {
            'global': ('global',),
            'agents': ('agents',)}},
    )

    # initial state
    initial_state = {
        'agents': {
            agent_id: {
                'external': {'A': 1},
                'trigger': False}}}

    sim = Engine(
        composite=composite,
        initial_state=initial_state
        )
    sim.update(time_total)
    output = sim.emitter.get_timeseries()

    assert len(output['agents']['1']['dead']) == time_dead + 1
    assert len(output['time']) == time_total + 1

    return output


def run_remove():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    os.makedirs(out_dir, exist_ok=True)
    output = test_remove()
    pp(output)


if __name__ == '__main__':
    run_remove()
