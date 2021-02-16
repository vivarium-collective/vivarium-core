from vivarium.core.process import Deriver
from vivarium.library.units import units


class MassToConcentration(Deriver):
    """ Adapts mass variable to mass concentration """
    name = 'mass_to_concentration'
    defaults = {
        'input_mass_units': 1.0 * units.fg,
        'concentration_unit': 1.0 * units.mmolar,
        'default_volume': 1.0 * units.fL,
        'molecular_weights': {
            'mass': 1.0 * units.fg / units.molec},
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.parameters['volume_unit'] = self.parameters['default_volume'].units

    def initial_state(self, config=None):
        return self.default_state()

    def ports_schema(self):
        keys = list(self.parameters['molecular_weights'].keys())
        return {
            'global': {
                'volume': {
                    '_default': self.parameters['default_volume']}
            },
            'input': {
                key: {
                    '_default': self.parameters['input_mass_units'],
                } for key in keys
            },
            'output': {
                key: {
                    '_default': 1.0,
                    '_update': 'set',
                } for key in keys
            }
        }

    def next_update(self, timestep, states):
        masses = states['input']
        volume = states['global']['volume']

        # do conversion
        # Concentration = mass/molecular_weight/characteristic volume
        # Note: here we just set the scale, not the volume
        mass_species_conc = {
            mol_id: (mass / self.parameters['molecular_weights'][mol_id] /
                     volume).to(self.parameters['concentration_unit']).magnitude
            for mol_id, mass in masses.items()}

        return {'output': mass_species_conc}


class MassToCount(Deriver):
    """ Adapts mass variable to mass concentration """
    name = 'mass_to_count'
    defaults = {
        'input_mass_units': 1.0 * units.fg,
        'molecular_weights': {
            'mass': 1.0 * units.fg / units.molec},
    }

    def initial_state(self, config=None):
        return self.default_state()

    def ports_schema(self):
        keys = list(self.parameters['molecular_weights'].keys())
        return {
            'input': {
                key: {
                    '_default': 1.0 * units.fg,
                } for key in keys
            },
            'output': {
                key: {
                    '_default': 1.0,
                    '_update': 'set',
                } for key in keys
            }
        }

    def next_update(self, timestep, states):
        masses = states['input']

        # do conversion
        # count = mass/molecular_weight
        # Note: here we just set the scale, not the volume
        mass_species_count = {
            mol_id: (mass / self.parameters[
                'molecular_weights'][mol_id]).magnitude
            for mol_id, mass in masses.items()}

        return {'output': mass_species_count}


def test_derivers():
    config = {
        'molecular_weights': {
            'A': 1.0 * units.fg / units.molec,
            'B': 2.0 * units.fg / units.molec,
        }}

    # MassToCount
    m_to_c = MassToCount(config)

    # convert mass to counts
    mass_in = m_to_c.initial_state()
    mass_in['input'] = {'A': 1 * units.fg, 'B': 1 * units.fg}
    counts_out = m_to_c.next_update(0, mass_in)

    # MassToConcentration
    m_to_conc = MassToConcentration(config)

    # convert mass to concentration
    mass_in = m_to_conc.initial_state()
    mass_in['input'] = {'A': 1 * units.fg, 'B': 1 * units.fg}
    concs_out = m_to_conc.next_update(0, mass_in)
    
    # asserts
    assert counts_out == {'output': {'A': 1.0, 'B': 0.5}}
    # TODO assert concs_out
    _ = concs_out

if __name__ == '__main__':
    test_derivers()
