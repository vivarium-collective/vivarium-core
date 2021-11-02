'''
=====
Datum
=====
'''
import copy

from typing import Any, Callable, Dict


class Datum(dict):
    '''
    The Datum class enables functions to be defined on dicts of a certain schema.
    It provides two class level variables:

    * `defaults`: a dictionary of keys to default values this Datum will have if
      none is provided to __init__
    * `schema`: a dictionary of keys to constructors which invoke subdata.

    Once these are defined, a Datum subclass can be constructed with a dict that provides any
    values beyond the defaults, and then all of the defined methods for that Datum subclass
    are available to operate on its values. Once the modifications are complete, it can be
    rendered back into a dict using the `to_dict()` method.
    '''

    schema: Dict[str, Callable] = {}
    defaults: Dict[str, Any] = {}

    def __init__(self, config):
        defaults = {
            key: value() if callable(value) else value
            for key, value in copy.deepcopy(self.defaults).items()}

        super().__init__(defaults)
        self.update(config)
        self.__dict__ = self

        for schema, realize in self.schema.items():
            if schema in self:
                value = self[schema]
                if isinstance(value, list):
                    value = [realize(item) for item in value]
                elif isinstance(value, dict):
                    value = {inner: realize(item) for inner, item in value.items()}
                else:
                    value = realize(value)
                self[schema] = value

    def to_dict(self):
        '''Convert the datum to a dictionary.'''
        return self

    def fields(self):
        '''Get the keys in the datum.'''
        return list(self.defaults.keys())

    def __repr__(self):
        return str(type(self)) + ': ' + str({
            key: value for key, value in self.items()})
