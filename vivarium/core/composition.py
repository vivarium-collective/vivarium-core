import os
from typing import Any, Dict, Optional
import uuid

from vivarium.composites.toys import (
    ToyCompartment,
    ToyLinearGrowthDeathProcess,
)
from vivarium.core.experiment import Experiment
from vivarium.core.process import (
    generate_derivers,
)
from vivarium.library.dict_utils import (
    deep_merge,
    deep_merge_check,
)

from vivarium.processes.timeline import TimelineProcess
from vivarium.processes.nonspatial_environment import \
    NonSpatialEnvironment

REFERENCE_DATA_DIR = os.path.join('vivarium', 'reference_data')
BASE_OUT_DIR = 'out'
TEST_OUT_DIR = os.path.join(BASE_OUT_DIR, 'tests')
PROCESS_OUT_DIR = os.path.join(BASE_OUT_DIR, 'processes')
COMPARTMENT_OUT_DIR = os.path.join(BASE_OUT_DIR, 'compartments')
COMPOSITE_OUT_DIR = os.path.join(BASE_OUT_DIR, 'composites')
EXPERIMENT_OUT_DIR = os.path.join(BASE_OUT_DIR, 'experiments')

FACTORY_KEY = '_factory'


# loading functions for compartment_hierarchy_experiment

def add_processes_to_hierarchy(
        processes,
        topology,
        factory_type,
        factory_config: Optional[Dict[str, Any]] = None,
        factory_topology: Optional[Dict[str, Any]] = None
):
    """ Use a factory to add processes and topology """
    factory_config = factory_config or {}
    factory_topology = factory_topology or {}

    # generate
    composite = factory_type(factory_config)
    network = composite.generate()
    new_processes = network['processes']
    new_topology = network['topology']

    # replace process names that already exist
    replace_name = []
    for name in new_processes.keys():
        if name in processes:
            replace_name.append(name)
    for name in replace_name:
        new_name = name + '_' + str(uuid.uuid1())
        new_processes[new_name] = new_processes[name]
        new_topology[new_name] = new_topology[name]
        del new_processes[name]
        del new_topology[name]

    # extend processes and topology list
    new_topology = deep_merge(new_topology, factory_topology)
    deep_merge(topology, new_topology)
    deep_merge_check(processes, new_processes)

    return processes, topology


def initialize_hierarchy(hierarchy):
    """ Make a hierarchy with initialized processes """
    processes = {}
    topology = {}
    for key, level in hierarchy.items():
        if key == FACTORY_KEY:
            if isinstance(level, list):
                for generator_def in level:
                    add_processes_to_hierarchy(
                        processes=processes,
                        topology=topology,
                        factory_type=generator_def['type'],
                        factory_config=generator_def.get(
                            'config', {}),
                        factory_topology=generator_def.get(
                            'topology', {}))

            elif isinstance(level, dict):
                add_processes_to_hierarchy(
                    processes=processes,
                    topology=topology,
                    factory_type=level['type'],
                    factory_config=level.get('config', {}),
                    factory_topology=level.get('topology', {}))
        else:
            network = initialize_hierarchy(level)
            deep_merge_check(processes, {key: network['processes']})
            deep_merge(topology, {key: network['topology']})

    return {
        'processes': processes,
        'topology': topology}


# list of keys expected in experiment settings
experiment_config_keys = [
        'experiment_id',
        'experiment_name',
        'description',
        'initial_state',
        'emitter',
        'emit_step',
        'display_info',
        'progress_bar',
        'invoke',
    ]


def compose_experiment(
        hierarchy=None,
        settings=None,
        initial_state=None,
):
    """Make an experiment with arbitrarily embedded compartments.

    Arguments:
        hierarchy: an embedded dictionary mapping the desired topology
          of nodes, with factories declared under a global FACTORY_KEY
          that maps to a dictionary with 'type', 'config', and
          'topology' for the processes in the Factory. Factories
          include lone processes.
        settings: experiment configuration settings.
        initial_state: is the initial_state.

    Returns:
        The experiment.
    """
    if settings is None:
        settings = {}
    if initial_state is None:
        initial_state = {}

    # make the hierarchy
    network = initialize_hierarchy(hierarchy)
    processes = network['processes']
    topology = network['topology']

    experiment_config = {
        'processes': processes,
        'topology': topology,
        'initial_state': initial_state}

    for key, setting in settings.items():
        if key in experiment_config_keys:
            experiment_config[key] = setting
    return Experiment(experiment_config)


# basic loading functions

def process_in_experiment(
        process,
        settings=None,
        initial_state=None,
):
    if settings is None:
        settings = {}
    if initial_state is None:
        initial_state = {}

    timeline = settings.get('timeline', {})
    environment = settings.get('environment', {})
    paths = settings.get('topology', {})

    processes = {'process': process}
    topology = {
        'process': {
            port: paths.get(port, (port,))
            for port in process.ports_schema().keys()}}

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

    # initialize the experiment
    experiment_config = {
        'processes': processes,
        'topology': topology,
        'initial_state': initial_state}
    for key, setting in settings.items():
        if key in experiment_config_keys:
            experiment_config[key] = setting
    return Experiment(experiment_config)


def compartment_in_experiment(
        compartment,
        settings=None,
        initial_state=None,
):
    if settings is None:
        settings = {}
    if initial_state is None:
        initial_state = {}

    compartment_config = settings.get('compartment', {})
    timeline = settings.get('timeline')
    environment = settings.get('environment')
    outer_path = settings.get('outer_path', tuple())

    network = compartment.generate(compartment_config, outer_path)
    processes = network['processes']
    topology = network['topology']

    if timeline is not None:
        # Add a timeline  requires the timeline argument in
        # settings to have a 'timeline' key. An optional 'paths'
        # key overrides the topology mapping from {port: path}.
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

    # initialize the experiment
    experiment_config = {
        'processes': processes,
        'topology': topology,
        'initial_state': initial_state}
    for key, setting in settings.items():
        if key in experiment_config_keys:
            experiment_config[key] = setting
    return Experiment(experiment_config)



# simulation functions

def simulate_process(
        process,
        settings: Optional[Dict[str, Any]] = None
):
    settings = settings or {}
    experiment = process_in_experiment(process, settings)
    return simulate_experiment(experiment, settings)


def simulate_process_in_experiment(
        process,
        settings: Optional[Dict[str, Any]] = None):
    settings = settings or {}
    experiment = process_in_experiment(process, settings)
    return simulate_experiment(experiment, settings)


def simulate_compartment_in_experiment(
        compartment,
        settings: Optional[Dict[str, Any]] = None):
    settings = settings or {}
    experiment = compartment_in_experiment(compartment, settings)
    return simulate_experiment(experiment, settings)


def simulate_experiment(
        experiment,
        settings: Optional[Dict[str, Any]] = None):
    '''
    run an experiment simulation
        Requires:
        - a configured experiment

    Returns:
        - a timeseries of variables from all ports.
        - if 'return_raw_data' is True, it returns the raw data instead
    '''
    settings = settings or {}
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
    return experiment.emitter.get_timeseries()



# Tests

class TestSimulateProcess:

    def test_process_deletion(self):
        '''Check that processes are successfully deleted'''
        process = ToyLinearGrowthDeathProcess({'targets': ['process']})
        settings = {
            'emit_step': 1,
            'topology': {
                'global': ('global',),
                'targets': tuple()}}

        output = simulate_process(process, settings)
        expected_masses = [
            # Mass stops increasing the iteration after mass > 5 because
            # cell dies
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0]
        masses = output['global']['mass']
        assert masses == expected_masses


# Toy processes


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
