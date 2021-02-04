"""
=============
Clock Process
=============
"""
from vivarium.core.process import Process

class Clock(Process):
    """ Clock

    A process for tracking global time
    """
    name = 'clock'
    defaults = {
        'time_step': 1.0}

    def initial_state(self, config=None):
        return {}

    def ports_schema(self):
        return {
            'global_time': {
                '_default': 0.0,
                '_updater': 'accumulate'}}

    def next_update(self, timestep, states):
        return {'global_time': timestep}
