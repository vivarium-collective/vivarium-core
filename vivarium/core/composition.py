from __future__ import absolute_import, division, print_function

import copy
import csv
import os
import io
import uuid

import numpy as np

from vivarium.core.experiment import Experiment
from vivarium.core.process import (
    Process,
    Deriver,
    Generator,
    generate_derivers,
)
from vivarium.core import emitter as emit
from vivarium.library.dict_utils import (
    deep_merge,
    deep_merge_check,
    flatten_timeseries,
    get_path_list_from_dict,
)
from vivarium.library.units import units

from vivarium.processes.timeline import TimelineProcess
from vivarium.processes.nonspatial_environment import NonSpatialEnvironment

REFERENCE_DATA_DIR = os.path.join('vivarium', 'reference_data')
TEST_OUT_DIR = os.path.join('out', 'tests')
PROCESS_OUT_DIR = os.path.join('out', 'processes')
COMPARTMENT_OUT_DIR = os.path.join('out', 'composites')
COMPOSITE_OUT_DIR = os.path.join('out', 'composites')
EXPERIMENT_OUT_DIR = os.path.join('out', 'experiments')


# loading functions
def make_agents(
        agent_ids,
        compartment,
        config=None
):
    """ Generate agents for each id
    Arguments:
    * **agent_ids**: list of agent ids
    * **compartment**: the compartment of the agent type
    * **config**: comparment configuration
    Returns:
        the intialized agent processes and topology
    """
    if config is None:
        config = {}
    processes = {}
    topology = {}
    for agent_id in agent_ids:
        agent_config = copy.deepcopy(config)
        agent = compartment.generate(dict(
            agent_config,
            agent_id=agent_id))

        # save processes and topology
        processes[agent_id] = agent['processes']
        topology[agent_id] = agent['topology']

    return {
        'processes': processes,
        'topology': topology}


def make_agent_ids(agents_config):
    """ Add agent ids to an agent config """
    agent_ids = []
    remove = []
    for idx, config in enumerate(agents_config):
        number = config.get('number', 1)
        if number < 1:
            remove.append(idx)
            continue
        if 'name' in config:
            name = config['name']
            if number > 1:
                new_agent_ids = [name + '_' + str(num) for num in range(number)]
            else:
                new_agent_ids = [name]
        else:
            new_agent_ids = [str(uuid.uuid1()) for num in range(number)]
        config['ids'] = new_agent_ids
        agent_ids.extend(new_agent_ids)
    # remove configs with number = 0
    for index in sorted(remove, reverse=True):
        del agents_config[index]
    return agent_ids


def add_process_to_tree(process_def, processes, topology):
    process_type = process_def['type']
    process_config = process_def['config']
    process_topology = process_def['topology']

    # make the process
    process = process_type(process_config)

    # extend processes and topology list
    name = process_def.get('name', process.name)
    deep_merge(processes, {name: process})
    deep_merge(topology, {name: process_topology})


def add_generator_to_tree(generator_def, processes, topology):
    generator_type = generator_def['type']
    generator_config = generator_def['config']

    # generate
    composite = generator_type(generator_config)
    network = composite.generate()
    new_processes = network['processes']
    new_topology = network['topology']

    # replace process names that already exist
    replace_name = []
    for name, p in new_processes.items():
        if name in processes:
            replace_name.append(name)
    for name in replace_name:
        new_name = name + '_' + str(uuid.uuid1())
        new_processes[new_name] = new_processes[name]
        new_topology[new_name] = new_topology[name]
        del new_processes[name]
        del new_topology[name]

    # extend processes and topology list
    composite_name = generator_def.get('name', composite.name)
    deep_merge_check(processes, {composite_name: new_processes})
    deep_merge(topology, {composite_name: new_topology})


def initialize_hierarchy(hierarchy):
    processes = {}
    topology = {}
    for key, level in hierarchy.items():
        if key == 'processes':
            if isinstance(level, list):
                for process_def in level:
                    add_process_to_tree(process_def, processes, topology)
            elif isinstance(level, dict):
                add_process_to_tree(level, processes, topology)
        elif key == 'generators':
            if isinstance(level, list):
                for generator_def in level:
                    add_generator_to_tree(generator_def, processes, topology)
            elif isinstance(level, dict):
                add_generator_to_tree(level, processes, topology)
        else:
            network = initialize_hierarchy(level)
            deep_merge_check(processes, {key: network['processes']})
            deep_merge(topology, {key: network['topology']})

    return {
        'processes': processes,
        'topology': topology}


def compartment_hierarchy_experiment(
        hierarchy=None,
        settings=None,
        initial_state=None,
        invoke=None,
):
    """Make an experiment with arbitrarily embedded compartments.

    Arguments:
        hierarchy: an embedded dictionary mapping the desired topology,
          with processes at a given level declared with a processes key
          that maps to a list of process configurations, and generators
          under a generators key mapping to a list of generator configurations.
        settings: settings include **emitter**.
        initial_state: is the initial_state.
        invoke: is the invoke object for calling updates.

    Returns:
        The experiment.
    """
    if settings is None:
        settings = {}
    if initial_state is None:
        initial_state = {}

    # experiment settings
    emitter = settings.get('emitter', {'type': 'timeseries'})

    # make the hierarchy
    network = initialize_hierarchy(hierarchy)
    processes = network['processes']
    topology = network['topology']

    experiment_config = {
        'processes': processes,
        'topology': topology,
        'emitter': emitter,
        'initial_state': initial_state}

    if settings.get('experiment_name'):
        experiment_config['experiment_name'] = settings.get('experiment_name')
    if settings.get('description'):
        experiment_config['description'] = settings.get('description')
    if invoke:
        experiment_config['invoke'] = invoke
    if 'emit_step' in settings:
        experiment_config['emit_step'] = settings['emit_step']
    return Experiment(experiment_config)


def agent_environment_experiment(
        agents_config=None,
        environment_config=None,
        initial_state=None,
        initial_agent_state=None,
        settings=None,
        invoke=None,
):
    """Make an experiment with agents placed in an environment under an `agents` store.

    Arguments:
        agents_config: the configuration for the agents
        environment_config: the configuration for the environment
        initial_state: the initial state for the hierarchy, with
            environment at the top level.
        initial_agent_state: the initial_state for agents, set under each agent_id.
        settings: settings include **emitter** and **agent_names**.
        invoke: is the invoke object for calling updates.

    Returns:
        The experiment.
    """
    if settings is None:
        settings = {}
    if initial_state is None:
        initial_state = {}

    # experiment settings
    emitter = settings.get('emitter', {'type': 'timeseries'})

    # initialize the agents
    if isinstance(agents_config, dict):
        # dict with single agent config
        agent_type = agents_config['type']
        agent_ids = agents_config['ids']
        agent_compartment = agent_type(agents_config['config'])
        agents = make_agents(agent_ids, agent_compartment, agents_config['config'])

        if initial_agent_state:
            initial_state['agents'] = {
                agent_id: initial_agent_state
                for agent_id in agent_ids}

    elif isinstance(agents_config, list):
        # list with multiple agent configurations
        agents = {
            'processes': {},
            'topology': {}}
        for config in agents_config:
            agent_type = config['type']
            agent_ids = config['ids']
            agent_compartment = agent_type(config['config'])
            new_agents = make_agents(agent_ids, agent_compartment, config['config'])
            deep_merge(agents['processes'], new_agents['processes'])
            deep_merge(agents['topology'], new_agents['topology'])

            if initial_agent_state:
                if 'agents' not in initial_state:
                    initial_state['agents'] = {}
                initial_state['agents'].update({
                    agent_id: initial_agent_state
                    for agent_id in agent_ids})

    if 'agents' in initial_state and 'diffusion' in environment_config['config']:
        environment_config[
            'config']['diffusion']['agents'] = initial_state['agents']

    # initialize the environment
    environment_type = environment_config['type']
    environment_compartment = environment_type(environment_config['config'])

    # combine processes and topologies
    network = environment_compartment.generate()
    processes = network['processes']
    topology = network['topology']
    processes['agents'] = agents['processes']
    topology['agents'] = agents['topology']

    if settings.get('agent_names') is True:
        # add an AgentNames processes, which saves the current agent names
        # to as store at the top level of the hierarchy
        processes['agent_names'] = AgentNames({})
        topology['agent_names'] = {
            'agents': ('agents',),
            'names': ('names',)
        }

    experiment_config = {
        'processes': processes,
        'topology': topology,
        'emitter': emitter,
        'initial_state': initial_state,
    }
    if settings.get('experiment_name'):
        experiment_config['experiment_name'] = settings.get('experiment_name')
    if settings.get('description'):
        experiment_config['description'] = settings.get('description')
    if invoke:
        experiment_config['invoke'] = invoke
    if 'emit_step' in settings:
        experiment_config['emit_step'] = settings['emit_step']
    return Experiment(experiment_config)

def process_in_compartment(
        process,
        topology={}
):
    """ put a lone process in a compartment"""
    class ProcessCompartment(Generator):
        def __init__(self, config):
            super(ProcessCompartment, self).__init__(config)
            self.schema_override = {}
            self.topology = topology
            self.process = process(self.config)

        def generate_processes(self, config):
            return {'process': self.process}

        def generate_topology(self, config):
            return {
                'process': {
                    port: self.topology.get(port, (port,)) for port in self.process.ports_schema().keys()}}

    return ProcessCompartment

def make_experiment_from_configs(
        agents_config={},
        environment_config={},
        initial_state={},
        settings={},
):
    # experiment settings
    emitter = settings.get('emitter', {'type': 'timeseries'})

    # initialize the agents
    agent_type = agents_config['agent_type']
    agent_ids = agents_config['agent_ids']
    agent = agent_type(agents_config['config'])
    agents = make_agents(agent_ids, agent, agents_config['config'])

    # initialize the environment
    environment_type = environment_config['environment_type']
    environment = environment_type(environment_config['config'])

    return make_experiment_from_compartments(
        environment.generate({}), agents, emitter, initial_state)

def make_experiment_from_compartment_dicts(
        environment_dict,
        agents_dict,
        emitter_dict,
        initial_state
):
    # environment_dict comes from environment.generate()
    # agents_dict comes from make_agents
    processes = environment_dict['processes']
    topology = environment_dict['topology']
    processes['agents'] = agents_dict['processes']
    topology['agents'] = agents_dict['topology']
    return Experiment({
        'processes': processes,
        'topology': topology,
        'emitter': emitter_dict,
        'initial_state': initial_state})

def process_in_experiment(
        process,
        settings={}
):
    initial_state = settings.get('initial_state', {})
    emitter = settings.get('emitter', {'type': 'timeseries'})
    emit_step = settings.get('emit_step')
    timeline = settings.get('timeline', [])
    environment = settings.get('environment', {})
    paths = settings.get('topology', {})

    processes = {'process': process}
    topology = {
        'process': {
            port: paths.get(port, (port,)) for port in process.ports_schema().keys()}}

    if timeline:
        # Adding a timeline to a process requires the timeline argument
        # in settings to have a 'timeline' key. An optional 'paths' key
        # overrides the topology mapping from {port: path}.
        timeline_process = TimelineProcess(timeline)
        timeline_paths = timeline.get('paths', {})
        processes.update({'timeline_process': timeline_process})
        timeline_ports = {
            port: timeline_paths.get(port, (port,))
            for port in timeline_process.ports()}
        topology.update({'timeline_process': timeline_ports})

    if environment:
        # Environment requires ports for external, fields, dimensions,
        # and global (for location)
        ports = environment.get(
            'ports',
            {
                'external': ('external',),
                'fields': ('fields',),
                'dimensions': ('dimensions',),
                'global': ('global',),
            }
        )
        environment_process = NonSpatialEnvironment(environment)
        processes.update({'environment_process': environment_process})
        topology.update({
            'environment_process': {
                'external': ports['external'],
                'fields': ports['fields'],
                'dimensions': ports['dimensions'],
                'global': ports['global'],
            },
        })

    # add derivers
    derivers = generate_derivers(processes, topology)
    processes = deep_merge(processes, derivers['processes'])
    topology = deep_merge(topology, derivers['topology'])

    return Experiment({
        'processes': processes,
        'topology': topology,
        'emitter': emitter,
        'emit_step': emit_step,
        'initial_state': initial_state})

def compartment_in_experiment(
        compartment,
        settings={}
):
    compartment_config = settings.get('compartment', {})
    timeline = settings.get('timeline')
    environment = settings.get('environment')
    outer_path = settings.get('outer_path', tuple())
    emit_step = settings.get('emit_step')

    network = compartment.generate(compartment_config, outer_path)
    processes = network['processes']
    topology = network['topology']

    if timeline is not None:
        # Adding a timeline to a compartment requires the timeline argument
        # in settings to have a 'timeline' key. An optional 'paths' key
        # overrides the topology mapping from {port: path}.
        timeline_process = TimelineProcess(timeline)
        timeline_paths = timeline.get('paths', {})
        processes.update({'timeline_process': timeline_process})
        timeline_ports = {
            port: timeline_paths.get(port, (port,))
            for port in timeline_process.ports()}
        topology.update({'timeline_process': timeline_ports})

    if environment is not None:
        # Environment requires ports for external, fields, dimensions,
        # and global (for location)
        ports = environment.get(
            'ports',
            {
                'external': ('external',),
                'fields': ('fields',),
                'dimensions': ('dimensions',),
                'global': ('global',),
            }
        )
        environment_process = NonSpatialEnvironment(environment)
        processes.update({'environment_process': environment_process})
        topology.update({
            'environment_process': {
                'external': ports['external'],
                'fields': ports['fields'],
                'dimensions': ports['dimensions'],
                'global': ports['global'],
            },
        })

    return Experiment({
        'processes': processes,
        'topology': topology,
        'emitter': settings.get('emitter', {'type': 'timeseries'}),
        'emit_step': emit_step,
        'initial_state': settings.get('initial_state', {})})


# simulation functions
def simulate_process(process, settings={}):
    experiment = process_in_experiment(process, settings)
    return simulate_experiment(experiment, settings)

def simulate_process_in_experiment(process, settings={}):
    experiment = process_in_experiment(process, settings)
    return simulate_experiment(experiment, settings)

def simulate_compartment_in_experiment(compartment, settings={}):
    experiment = compartment_in_experiment(compartment, settings)
    return simulate_experiment(experiment, settings)

def simulate_experiment(experiment, settings={}):
    '''
    run an experiment simulation
        Requires:
        - a configured experiment

    Returns:
        - a timeseries of variables from all ports.
        - if 'return_raw_data' is True, it returns the raw data instead
    '''
    total_time = settings.get('total_time', 10)
    return_raw_data = settings.get('return_raw_data', False)

    if 'timeline' in settings:
        all_times = [t[0] for t in settings['timeline']['timeline']]
        total_time = max(all_times)

    # run simulation
    experiment.update(total_time)
    experiment.end()

    # return data from emitter
    if return_raw_data:
        return experiment.emitter.get_data()
    else:
        return experiment.emitter.get_timeseries()


# timeseries functions
def agent_timeseries_from_data(data, agents_key='cells'):
    timeseries = {}
    for time, all_states in data.items():
        agent_data = all_states[agents_key]
        for agent_id, ports in agent_data.items():
            if agent_id not in timeseries:
                timeseries[agent_id] = {}
            for port_id, states in ports.items():
                if port_id not in timeseries[agent_id]:
                    timeseries[agent_id][port_id] = {}
                for state_id, state in states.items():
                    if state_id not in timeseries[agent_id][port_id]:
                        timeseries[agent_id][port_id][state_id] = []
                    timeseries[agent_id][port_id][state_id].append(state)
    return timeseries

def save_timeseries(timeseries, out_dir='out'):
    flattened = flatten_timeseries(timeseries)
    save_flat_timeseries(flattened, out_dir)

def save_flat_timeseries(timeseries, out_dir='out'):
    '''Save a timeseries as a CSV in out_dir'''
    rows = np.transpose(list(timeseries.values())).tolist()
    with open(os.path.join(out_dir, 'simulation_data.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(timeseries.keys())
        writer.writerows(rows)

def load_timeseries(path_to_csv):
    '''Load a timeseries saved as a CSV using save_timeseries.

    The timeseries is returned in flattened form.
    '''
    with io.open(path_to_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        timeseries = {}
        for row in reader:
            for header, elem in row.items():
                if elem == '':
                    elem = None
                if elem is not None:
                    elem = float(elem)
                timeseries.setdefault(header, []).append(elem)
    return timeseries

def timeseries_to_ndarrays(timeseries, keys=None):
    '''After filtering by keys, convert timeseries to dict of ndarrays

    Returns:
        dict: Mapping from timeseries variables to an ndarray of the
            variable values.
    '''
    if keys is None:
        keys = timeseries.keys()
    return {
        key: np.array(timeseries[key], dtype=np.float) for key in keys}

def _prepare_timeseries_for_comparison(
    timeseries1, timeseries2, keys=None,
    required_frac_checked=0.9,
):
    '''Prepare two timeseries for comparison

    Arguments:
        timeseries1: One timeseries. Must be flattened and include times
            under the 'time' key.
        timeseries2: The other timeseries. Same requirements as
            timeseries1.
        keys: Keys of the timeseries whose values will be checked for
            correlation. If not specified, all keys present in both
            timeseries are used.
        required_frac_checked: The required fraction of timepoints in a
            timeseries that must be checked. If this requirement is not
            satisfied, which might occur if the two timeseries share few
            timepoints, the test wll fail.

    Returns:
        A tuple of an ndarray for each of the two timeseries and a list of
        the keys for the rows of the arrays. Each ndarray has a row for
        each key, in the order of keys. The ndarrays have only the
        columns corresponding to the timepoints common to both
        timeseries.

    Raises:
        AssertionError: If a correlation is strictly below the
            threshold or if too few timepoints are common to both
            timeseries.
    '''
    if 'time' not in timeseries1 or 'time' not in timeseries2:
        raise AssertionError('Both timeseries must have key "time"')
    if keys is None:
        keys = set(timeseries1.keys()) & set(timeseries2.keys())
    else:
        if 'time' not in keys:
            keys.append('time')
    keys = list(keys)
    time_index = keys.index('time')
    shared_times = set(timeseries1['time']) & set(timeseries2['time'])
    frac_timepoints_checked = (
        len(shared_times)
        / min(len(timeseries1['time']), len(timeseries2['time']))
    )
    if frac_timepoints_checked < required_frac_checked:
        raise AssertionError(
            'The timeseries share too few timepoints: '
            '{} < {}'.format(
                frac_timepoints_checked, required_frac_checked)
        )
    masked = []
    for ts in (timeseries1, timeseries2):
        arrays_dict = timeseries_to_ndarrays(ts, keys)
        arrays_dict_shared_times = {}
        for key, array in arrays_dict.items():
            # Filters out times after data ends
            times_for_array = arrays_dict['time'][:len(array)]
            arrays_dict_shared_times[key] = array[
                np.isin(times_for_array, list(shared_times))]
        masked.append(arrays_dict_shared_times)
    return (
        masked[0],
        masked[1],
        keys,
    )

def assert_timeseries_correlated(
    timeseries1, timeseries2, keys=None,
    default_threshold=(1 - 1e-10), thresholds={},
    required_frac_checked=0.9,
):
    '''Check that two timeseries are correlated.

    Uses a Pearson correlation coefficient. Only the data from
    timepoints common to both timeseries are compared.

    Arguments:
        timeseries1: One timeseries. Must be flattened and include times
            under the 'time' key.
        timeseries2: The other timeseries. Same requirements as
            timeseries1.
        keys: Keys of the timeseries whose values will be checked for
            correlation. If not specified, all keys present in both
            timeseries are used.
        default_threshold: The threshold correlation coefficient to use
            when a threshold is not specified in thresholds.
        thresholds: Dictionary of key-value pairs where the key is a key
            in both timeseries and the value is the threshold
            correlation coefficient to use when checking that key
        required_frac_checked: The required fraction of timepoints in a
            timeseries that must be checked. If this requirement is not
            satisfied, which might occur if the two timeseries share few
            timepoints, the test wll fail. This is also the fraction of
            timepoints for each variable that must be non-nan in both
            timeseries. Note that the denominator of this fraction is
            the number of shared timepoints that are non-nan in either
            of the timeseries.

    Raises:
        AssertionError: If a correlation is strictly below the
            threshold or if too few timepoints are common to both
            timeseries.
    '''
    arrays1, arrays2, keys = _prepare_timeseries_for_comparison(
        timeseries1, timeseries2, keys, required_frac_checked)
    for key in keys:
        both_nan = np.isnan(arrays1[key]) & np.isnan(arrays2[key])
        valid_indices = ~(
            np.isnan(arrays1[key]) | np.isnan(arrays2[key]))
        frac_checked = valid_indices.sum() / (~both_nan).sum()
        if frac_checked < required_frac_checked:
            raise AssertionError(
                'Timeseries share too few non-nan values for variable '
                '{}: {} < {}'.format(
                    key, frac_checked, required_frac_checked
                )
            )
        corrcoef = np.corrcoef(
            arrays1[key][valid_indices],
            arrays2[key][valid_indices],
        )[0][1]
        threshold = thresholds.get(key, default_threshold)
        if corrcoef < threshold:
            raise AssertionError(
                'The correlation coefficient for '
                '{} is too small: {} < {}'.format(
                    key, corrcoef, threshold)
            )

def assert_timeseries_close(
    timeseries1, timeseries2, keys=None,
    default_tolerance=(1 - 1e-10), tolerances={},
    required_frac_checked=0.9,
):
    '''Check that two timeseries are similar.

    Ensures that each pair of data points between the two timeseries are
    within a tolerance of each other, after filtering out timepoints not
    common to both timeseries.

    Arguments:
        timeseries1: One timeseries. Must be flattened and include times
            under the 'time' key.
        timeseries2: The other timeseries. Same requirements as
            timeseries1.
        keys: Keys of the timeseries whose values will be checked for
            correlation. If not specified, all keys present in both
            timeseries are used.
        default_tolerance: The tolerance to use when not specified in
            tolerances.
        tolerances: Dictionary of key-value pairs where the key is a key
            in both timeseries and the value is the tolerance to use
            when checking that key.
        required_frac_checked: The required fraction of timepoints in a
            timeseries that must be checked. If this requirement is not
            satisfied, which might occur if the two timeseries share few
            timepoints, the test wll fail.

    Raises:
        AssertionError: If a pair of data points have a difference
            strictly above the tolerance threshold or if too few
            timepoints are common to both timeseries.
    '''
    arrays1, arrays2, keys = _prepare_timeseries_for_comparison(
        timeseries1, timeseries2, keys, required_frac_checked)
    for key in keys:
        tolerance = tolerances.get(key, default_tolerance)
        close_mask = np.isclose(arrays1[key], arrays2[key],
            atol=tolerance, equal_nan=True)
        if not np.all(close_mask):
            print('Timeseries 1:', arrays1[key][~close_mask])
            print('Timeseries 2:', arrays2[key][~close_mask])
            raise AssertionError(
                'The data for {} differed by more than {}'.format(
                    key, tolerance)
            )


# TESTS
class ToyLinearGrowthDeathProcess(Process):

    name = 'toy_linear_growth_death'

    GROWTH_RATE = 1.0
    THRESHOLD = 6.0

    def __init__(self, initial_parameters={}):
        self.targets = initial_parameters.get('targets')
        super(ToyLinearGrowthDeathProcess, self).__init__(initial_parameters)

    def ports_schema(self):
        return {
            'global': {
                'mass': {
                    '_default': 1.0,
                    '_emit': True}},
            'targets': {
                target: {
                    '_default': None}
                for target in self.targets}}

    def next_update(self, timestep, states):
        mass = states['global']['mass']
        mass_grown = (
            ToyLinearGrowthDeathProcess.GROWTH_RATE * timestep)
        update = {
            'global': {'mass': mass_grown},
        }
        if mass > ToyLinearGrowthDeathProcess.THRESHOLD:
            update['global'] = {
                '_delete': [(target,) for target in self.targets]}

        return update


class TestSimulateProcess:

    def test_process_deletion(self):
        '''Check that processes are successfully deleted'''
        process = ToyLinearGrowthDeathProcess({'targets': ['process']})
        settings = {
            'emit_step': 1,
            'topology': {
                'global': ('global',),
                'targets': tuple()}}

        timeseries = simulate_process(process, settings)
        expected_masses = [
            # Mass stops increasing the iteration after mass > 5 because
            # cell dies
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0]
        masses = timeseries['global']['mass']
        assert masses == expected_masses


# toy processes
class ToyMetabolism(Process):
    name = 'toy_metabolism'

    def __init__(self, initial_parameters={}):
        parameters = {'mass_conversion_rate': 1}
        parameters.update(initial_parameters)
        super(ToyMetabolism, self).__init__(parameters)

    def ports_schema(self):
        ports = {
            'pool': ['GLC', 'MASS']}
        return {
            port_id: {
                key: {
                    '_default': 0.0,
                    '_emit': True}
                for key in keys}
            for port_id, keys in ports.items()}

    def next_update(self, timestep, states):
        update = {}
        glucose_required = timestep / self.parameters['mass_conversion_rate']
        if states['pool']['GLC'] >= glucose_required:
            update = {
                'pool': {
                    'GLC': -2,
                    'MASS': 1}}

        return update

class ToyTransport(Process):
    name = 'toy_transport'

    def __init__(self, initial_parameters={}):
        parameters = {'intake_rate': 2}
        parameters.update(initial_parameters)
        super(ToyTransport, self).__init__(parameters)

    def ports_schema(self):
        ports = {
            'external': ['GLC'],
            'internal': ['GLC']}
        return {
            port_id: {
                key: {
                    '_default': 0.0,
                    '_emit': True}
                for key in keys}
            for port_id, keys in ports.items()}

    def next_update(self, timestep, states):
        update = {}
        intake = timestep * self.parameters['intake_rate']
        if states['external']['GLC'] >= intake:
            update = {
                'external': {'GLC': -2, 'MASS': 1},
                'internal': {'GLC': 2}}

        return update

class ToyDeriveVolume(Deriver):
    name = 'toy_derive_volume'

    def __init__(self, initial_parameters={}):
        parameters = {}
        super(ToyDeriveVolume, self).__init__(parameters)

    def ports_schema(self):
        ports = {
            'compartment': ['MASS', 'DENSITY', 'VOLUME']}
        return {
            port_id: {
                key: {
                    '_updater': 'set' if key == 'VOLUME' else 'accumulate',
                    '_default': 0.0,
                    '_emit': True}
                for key in keys}
            for port_id, keys in ports.items()}

    def next_update(self, timestep, states):
        volume = states['compartment']['MASS'] / states['compartment']['DENSITY']
        update = {
            'compartment': {'VOLUME': volume}}

        return update

class ToyDeath(Process):
    name = 'toy_death'

    def __init__(self, initial_parameters={}):
        self.targets = initial_parameters.get('targets', [])
        super(ToyDeath, self).__init__({})

    def ports_schema(self):
        return {
            'compartment': {
                'VOLUME': {
                    '_default': 0.0,
                    '_emit': True}},
            'global': {
                target: {
                    '_default': None}
                for target in self.targets}}

    def next_update(self, timestep, states):
        volume = states['compartment']['VOLUME']
        update = {}

        if volume > 1.0:
            # kill the cell
            update = {
                'global': {
                    '_delete': [
                        (target,)
                        for target in self.targets]}}

        return update

class ToyCompartment(Generator):
    '''
    a toy compartment for testing

    '''
    def __init__(self, config):
        super(ToyCompartment, self).__init__(config)

    def generate_processes(self, config):
        return {
            'metabolism': ToyMetabolism(
                {'mass_conversion_rate': 0.5}), # example of overriding default parameters
            'transport': ToyTransport(),
            'death': ToyDeath({'targets': [
                'metabolism',
                'transport']}),
            'external_volume': ToyDeriveVolume(),
            'internal_volume': ToyDeriveVolume()
        }

    def generate_topology(self, config):
        return{
            'metabolism': {
                'pool': ('cytoplasm',)},
            'transport': {
                'external': ('periplasm',),
                'internal': ('cytoplasm',)},
            'death': {
                'global': tuple(),
                'compartment': ('cytoplasm',)},
            'external_volume': {
                'compartment': ('periplasm',)},
            'internal_volume': {
                'compartment': ('cytoplasm',)}}


def test_compartment():
    toy_compartment = ToyCompartment({})
    settings = {
        'total_time': 10,
        'initial_state': {
            'periplasm': {
                'GLC': 20,
                'MASS': 100,
                'DENSITY': 10},
            'cytoplasm': {
                'GLC': 0,
                'MASS': 3,
                'DENSITY': 10}}}
    return simulate_compartment_in_experiment(toy_compartment, settings)


if __name__ == '__main__':
    TestSimulateProcess().test_process_deletion()
    timeseries = test_compartment()
    print(timeseries)
