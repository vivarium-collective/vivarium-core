from vivarium.core.process import Deriver


class DivideCondition(Deriver):
    """ Divide Condition Process """
    name = 'divide_condition'
    defaults = {}

    def __init__(self, parameters=None):
        super(DivideCondition, self).__init__(parameters)
        self.threshold = self.parameters['threshold']

    def initial_state(self, config=None):
        return {}

    def ports_schema(self):
        return {
            'variable': {},
            'divide': {
                '_default': False,
                '_updater': 'set',
                '_divider': 'zero'}}

    def next_update(self, timestep, states):
        if states['variable'] >= self.threshold:
            return {'divide': True}
        else:
            return {}
