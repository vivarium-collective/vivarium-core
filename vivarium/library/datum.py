from typing import Any, Callable, Dict, List


def first(a_list: List):
    if a_list:
        return a_list[0]


def first_value(d: Dict):
    if d:
        return d[list(d.keys())[0]]  # TODO(jerry): iter(d.values()).__next__() would be faster


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
        super().__init__(self.defaults)
        self.update(config)
        self.__dict__ = self  # TODO(jerry): Instead define `def __getattr__(self, item): return self[item]`?

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
        return self

    def fields(self):
        return list(self.defaults.keys())

    def __repr__(self):
        return str(type(self)) + ': ' + str({
            key: value for key, value in self.items()})
