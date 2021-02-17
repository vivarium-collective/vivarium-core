"""
==================
StripUnits Deriver
==================
"""

from typing import Dict, Any

from vivarium.core.process import Deriver
from vivarium.library.units import remove_units


class StripUnits(Deriver):
    """StripUnits Deriver

    Reads values specified by the 'keys' parameter under the 'units' port,
    removes the units, and updates keys of the same name under the '_no_units'
    port. Converts values before stripping them for all {key: unit_target}
    pairs declared in the 'convert' parameter dictionary.
    """
    name = 'strip_units'
    defaults: Dict[str, Any] = {
        'keys': [],
        'convert': {}}

    def __init__(self, parameters):
        super().__init__(parameters)
        self.convert = self.parameters['convert']

    def ports_schema(self):
        return {
            'units': {key: {} for key in self.parameters['keys']},
            'no_units': {key: {} for key in self.parameters['keys']}}

    def next_update(self, timestep, states):
        converted_units = {
            state: value.to(self.convert[state])
            if state in self.convert else value
            for state, value in states['units'].items()}
        return {
            'no_units': {
                key: {
                    '_value': remove_units(value),
                    '_updater': 'set'
                } for key, value in converted_units.items()
            }}
