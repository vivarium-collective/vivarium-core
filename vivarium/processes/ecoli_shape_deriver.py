import copy
import math

from scipy import constants
import numpy as np

from vivarium.core.process import Deriver
from vivarium.library.units import units
from vivarium.library.dict_utils import deep_merge



PI = math.pi
AVOGADRO = constants.N_A * 1 / units.mol



def length_from_volume(volume, width):
    '''
    get cell length from volume, using the following equation for capsule volume, with V=volume, r=radius,
    a=length of cylinder without rounded caps, l=total length:

    V = (4/3)*PI*r^3 + PI*r^2*a
    l = a + 2*r
    '''
    radius = width / 2
    cylinder_length = (volume - (4/3) * PI * radius**3) / (PI * radius**2)
    total_length = cylinder_length + 2 * radius
    return total_length

def volume_from_length(length, width):
    '''
    inverse of length_from_volume
    '''
    radius = width / 2
    cylinder_length = length - width
    volume = cylinder_length * (PI * radius**2) + (4 / 3) * PI * radius**3
    return volume

def surface_area_from_length(length, width):
    '''
    SA = 3*PI*r^2 + 2*PI*r*a
    '''
    radius = width / 2
    cylinder_length = length - width
    surface_area = 3 * PI * radius**2 + 2 * PI * radius * cylinder_length
    return surface_area



class EcoliShape(Deriver):
    """
    Process for deriving volume, mmol_to_counts, and shape from the cell mass
    """

    name = 'globals_deriver'
    defaults = {
        'width': 1,  # um
        'initial_mass': 1400 * units.fg,  # wet mass in fg
        'density': 1100 * units.g / units.L,
        'periplasm_volume_fraction': 0.3,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def initial_state(self, config=None):
        volume = self.parameters['initial_mass'] / self.parameters['density']
        mmol_to_counts = (AVOGADRO * volume)
        return {
            'global': {
                'volume': volume.to('fL'),
                'mmol_to_counts': mmol_to_counts.to('L/mmol')}}

    def ports_schema(self):
        set_states = [
            'volume', 'mmol_to_counts', 'length', 'surface_area',
            'periplasm_volume',
        ]
        split_divide = [
            'volume', 'length', 'surface_area', 'periplasm_volume']
        emit = {'global': ['volume', 'width', 'length', 'surface_area']}

        # default state
        mass = self.parameters['initial_mass']
        width = self.parameters['width']
        density = self.parameters['density']
        volume = mass/density
        mmol_to_counts = (AVOGADRO * volume).to('L/mmol')
        length = length_from_volume(volume.magnitude, width)
        surface_area = surface_area_from_length(length, width)
        periplasm_volume = volume * self.parameters[
            'periplasm_volume_fraction']

        default_state = {
            'global': {
                'mass': mass,
                'volume': volume.to('fL'),
                'mmol_to_counts': mmol_to_counts,
                'density': density,
                'width': width,
                'length': length,
                'surface_area': surface_area,
                'periplasm_volume': periplasm_volume,
            },
        }

        schema = {}
        for port, states in default_state.items():
            schema[port] = {}
            for state_id, value in states.items():
                schema[port][state_id] = {}
                if state_id in set_states:
                    schema[port][state_id]['_updater'] = 'set'
                if state_id in emit[port]:
                    schema[port][state_id]['_emit'] = True
                if state_id in split_divide:
                    schema[port][state_id]['_divider'] = 'split'
                if state_id in default_state[port]:
                    schema[port][state_id]['_default'] = default_state[port][state_id]

        return schema

    def next_update(self, timestep, states):
        density = states['global']['density']
        mass = states['global']['mass'].to('fg')
        width = states['global']['width']

        # get volume from mass, and more variables from volume
        volume = mass / density
        mmol_to_counts = (AVOGADRO * volume).to('L/mmol')
        length = length_from_volume(volume.magnitude, width)
        surface_area = surface_area_from_length(length, width)
        periplasm_volume = volume * self.parameters[
            'periplasm_volume_fraction']

        return {
            'global': {
                'volume': volume.to('fL'),
                'mmol_to_counts': mmol_to_counts,
                'length': length,
                'surface_area': surface_area,
                'periplasm_volume': periplasm_volume,
            },
        }



def test_deriver(total_time=10):

    growth_rate = 6e-3

    # configure process
    deriver = EcoliShape({})
    state = deriver.default_state()

    # initialize saved data
    saved_state = {}

    ## simulation
    timeline = [(total_time, {})]
    time = 0
    timestep = 1  # sec
    saved_state[time] = state
    while time < timeline[-1][0]:
        time += timestep
        for (t, change_dict) in timeline:
            if time >= t:
                for key, change in change_dict.items():
                    state[key].update(change)

        # set mass, counts
        mass = state['global']['mass']
        state['global']['mass'] = mass * np.exp(growth_rate * timestep)

        # get update
        update = deriver.next_update(timestep, state)

        # set derived state
        state['global'].update(update['global'])

        # save state
        saved_state[time] = copy.deepcopy(state)

    return saved_state


if __name__ == '__main__':
    saved_data = test_deriver(100)
    print(saved_data)
