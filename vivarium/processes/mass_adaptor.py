from vivarium.core.process import Deriver
from vivarium.library.units import units


class CountsToConcentration(Deriver):
    """ Adapts counts to mass-based concentration """
    name = 'counts_to_concentration'
    defaults = {
        'concentration_unit': units.mg / units.mL,
        'default_volume': 1 * units.fL,
        # map molecule name to mw in units.g / units.mol
        'molecular_weights': {},
    }

    def __init__(self, parameters=None):
        parameters = parameters or {}
        if 'molecular_weights' not in parameters:
            parameters['molecular_weights'] = {
                'mass': 1.0 * units.g / units.mol}
        super().__init__(parameters)
        for mol_id, mw in self.parameters['molecular_weights'].items():
            try:
                mw.to(units.g / units.mol)
            except ValueError:
                print(
                    f"{mol_id} needs a mw convertable to units.g / units.mol")

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
                    '_default': 0,
                } for key in keys
            },
            'output': {
                key: {
                    '_default': 1.0 * self.parameters['concentration_unit'],
                    '_updater': 'set',
                } for key in keys
            }
        }

    def next_update(self, timestep, states):
        counts = states['input']
        volume = states['global']['volume']

        # do conversion
        # Concentration = mass/molecular_weight/characteristic volume
        # Note: here we just set the scale, not the volume
        mass_species_conc = {
            mol_id: (count * units.molec * self.parameters[
                'molecular_weights'][mol_id] / volume).to(
                self.parameters['concentration_unit'])
            for mol_id, count in counts.items()}

        return {'output': mass_species_conc}


class MassToCount(Deriver):
    """ Adapts mass variable to counts """
    name = 'mass_to_count'
    defaults = {
        'input_mass_units': 1.0 * units.fg,
        'molecular_weights': {},
    }

    def __init__(self, parameters=None):
        parameters = parameters or {}
        if 'molecular_weights' not in parameters:
            parameters['molecular_weights'] = {
                'mass': 1.0 * units.fg / units.molec}
        super().__init__(parameters)
        for mol_id, mw in self.parameters['molecular_weights'].items():
            try:
                mw.to(units.g / units.mol)
            except ValueError:
                print(
                    f"{mol_id} needs a mw convertable to units.g / units.mol")

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
                    '_updater': 'set',
                } for key in keys
            },
            'global': {
                'volume': {}
            }
        }

    def next_update(self, timestep, states):
        masses = states['input']

        # do conversion
        # count = mass/molecular_weight
        # Note: here we just set the scale, not the volume
        mass_species_count = {
            mol_id: int((mass / self.parameters[
                'molecular_weights'][mol_id]).magnitude)
            for mol_id, mass in masses.items()}

        return {'output': mass_species_count}



class MassToMolar(Deriver):
    """ Adapts mass variable to mmolar concentration """
    name = 'mass_to_concentration'
    defaults = {
        'mass_units': units.fg,
        'concentration_unit': units.mmolar,
        'default_volume': 1 * units.fL,
        # map molecule name to mw in units.g / units.mol
        'molecular_weights': {},
    }

    def __init__(self, parameters=None):
        parameters = parameters or {}
        if 'molecular_weights' not in parameters:
            parameters['molecular_weights'] = {
                'mass': 1.0 * units.g / units.mol}
        super().__init__(parameters)
        for mol_id, mw in self.parameters['molecular_weights'].items():
            try:
                mw.to(units.g / units.mol)
            except ValueError:
                print(
                    f"{mol_id} needs a mw convertable to units.g / units.mol")

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
                    '_default': 0 * self.parameters['mass_units'],
                } for key in keys
            },
            'output': {
                key: {
                    '_default': 1.0 * self.parameters['concentration_unit'],
                    '_updater': 'set',
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
                     volume).to(self.parameters['concentration_unit'])
            for mol_id, mass in masses.items()}

        return {'output': mass_species_conc}



def test_derivers():
    config = {
        'molecular_weights': {
            'A': 1.0 * units.g / units.mol,
            'B': 2.0 * units.g / units.mol,
        }}

    # MassToCount
    m_to_c = MassToCount(config)

    # convert mass to counts
    mass_in = m_to_c.initial_state()
    mass_in['input'] = {'A': 10 * units.fg, 'B': 10 * units.fg}
    counts_out = m_to_c.next_update(0, mass_in)

    # MassToConcentration
    m_to_conc = CountsToConcentration(config)

    # convert mass to concentration
    mass_in = m_to_conc.initial_state()
    mass_in['input'] = {'A': 10, 'B': 10}
    concs_out = m_to_conc.next_update(0, mass_in)

    # asserts
    assert counts_out == {'output': {'A': 10, 'B': 5}}
    # TODO assert concs_out
    _ = concs_out

if __name__ == '__main__':
    test_derivers()
