from __future__ import absolute_import, division, print_function

import numpy as np
from scipy import constants

from vivarium.library.units import units
from vivarium.core.process import Deriver

AVOGADRO = constants.N_A * 1 / units.mol


class NonSpatialEnvironment(Deriver):
    '''A non-spatial environment with volume'''

    name = 'nonspatial_environment'
    defaults = {
        'volume': 1e-12 * units.L,
        'concentrations': {},
    }

    def __init__(self, parameters=None):
        super(NonSpatialEnvironment, self).__init__(parameters)
        volume = parameters.get('volume', self.defaults['volume'])
        self.mmol_to_counts = (AVOGADRO.to('1/mmol') * volume).to('L/mmol')

    def ports_schema(self):
        bin_x = 1 * units.um
        bin_y = 1 * units.um
        depth = self.parameters['volume'] / bin_x / bin_y
        n_bin_x = 1
        n_bin_y = 1
        schema = {
            'external': {
                '*': {
                    '_value': 0,
                },
            },
            'fields': {
                '*': {
                    '_default': np.ones((1, 1)),
                },
            },
            'dimensions': {
                'depth': {
                    '_value': depth.to(units.um).magnitude,
                },
                'n_bins': {
                    '_value': [n_bin_x, n_bin_y],
                },
                'bounds': {
                    '_value': [
                        n_bin_x * bin_x.to(units.um).magnitude,
                        n_bin_y * bin_y.to(units.um).magnitude,
                    ],
                },
            },
            'global': {
                'location': {
                    '_value': [0.5, 0.5],
                },
                'volume': {
                    '_value': self.parameters['volume'],
                }
            },
        }
        # add field concentrations
        field_schema = {
            field_id: {
                '_value': np.array([[conc]])
            } for field_id, conc in self.parameters['concentrations'].items()}
        schema['fields'].update(field_schema)
        return schema

    def next_update(self, timestep, states):
        fields = states['fields']

        update = {
            'external': {
                mol_id: {
                    '_updater': 'set',
                    '_value': field[0][0],
                }
                for mol_id, field in fields.items()
            },
        }

        return update
