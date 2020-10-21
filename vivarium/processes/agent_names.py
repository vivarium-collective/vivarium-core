from __future__ import absolute_import, division, print_function

from vivarium.core.process import Deriver


class AgentNames(Deriver):

    name = 'agent_names'
    defaults = {}

    def __init__(self, initial_parameters=None):
        if initial_parameters is None:
            initial_parameters = {}
        super(AgentNames, self).__init__(initial_parameters)

    def ports_schema(self):
        return {
            'agents': {
                '*': {}
            },
            'names': {
                '_default': [],
                '_updater': 'set',
                '_emit': True,
            },
        }

    def next_update(self, timestep, states):
        agents = states['agents']
        return {'names': list(agents.keys())}
