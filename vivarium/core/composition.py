"""
===========
Composition
===========

Helper functions for initializing and running experiments.
"""

import warnings
warnings.warn(
    "composition.py is no longer being maintained and will be "
    "removed in a future version of vivarium-core. Please use "
    "Engine directly to run simulations, and Composite.merge "
    "for composition", DeprecationWarning)

# typing
from typing import (
    Any, Dict, Optional)
from vivarium.core.types import (
    Topology, HierarchyPath)

from vivarium.core.process import Process
from vivarium.core.composer import Composer, Composite
from vivarium.core.engine import Engine
from vivarium.core.directories import (
    REFERENCE_DATA_DIR, BASE_OUT_DIR, TEST_OUT_DIR,
    PROCESS_OUT_DIR, COMPARTMENT_OUT_DIR, COMPOSITE_OUT_DIR,
    EXPERIMENT_OUT_DIR,
)
from vivarium.library.dict_utils import deep_merge

from vivarium.processes.timeline import TimelineProcess
from vivarium.processes.nonspatial_environment import \
    NonSpatialEnvironment

# toys for testing
from vivarium.composites.toys import (
    PoQo,
    ToyLinearGrowthDeathProcess,
    ExchangeA,
    ToyCompartment
)


# list of keys expected in experiment settings
experiment_config_keys = [
        'experiment_id',
        'experiment_name',
        'description',
        'emitter',
        'emit_step',
        'display_info',
        'progress_bar',
        'invoke',
    ]


# experiment loading functions
def add_timeline(
        processes: Dict[str, Process],
        topology: Dict[str, Topology],
        timeline: Dict[str, Any],
) -> None:
    """Add a timeline process to a composite

    Args:
        processes: with ``{'process_name': Process}``
        topology (dict): with ``{'process_name': topology mapping}``
        timeline (dict): with ``timeline`` key. An optional ``paths``
            key overrides the topology mapping from (port: path).
    """
    timeline_paths = timeline.get('paths', {})
    timeline_process = TimelineProcess(timeline)
    processes.update({
        TimelineProcess.name: timeline_process})

    # add topology
    timeline_topology = {
        port: timeline_paths.get(port, (port,))
        for port in timeline_process.ports()}
    topology.update({TimelineProcess.name: timeline_topology})


def add_environment(
        processes: Dict[str, Any],
        topology: Dict[str, Any],
        environment: Dict[str, Any],
) -> None:
    """Add a NonSpatialEnvironment to a composite

    Args:
        processes: with ``{'process_name': Process}``
        topology: with ``{'process_name': topology mapping}``
        environment: with ``environment`` key. An optional ``paths`` key
            overrides the topology mapping from (port: path).
    """

    overide_topology = environment.get('paths', {})
    environment_process = NonSpatialEnvironment(environment)
    processes.update({
        environment_process.name: environment_process})

    # add topology
    environment_topology = environment_process.generate_topology({})[
        environment_process.name]
    environment_topology = deep_merge(
        environment_topology,
        overide_topology)
    topology.update({
        environment_process.name: environment_topology})


def process_in_experiment(
        process: Process,
        settings: Dict[str, Any] = None,
        initial_state: Dict[str, Any] = None,
) -> Engine:
    """Put a Process in an Engine

    Args:
        process: the Process to put into the Engine
        settings: a dictionary of optional configuration options,
            keywords include timeline, environment, and topology that
            add to or modify the Process.
        initial_state: initial state to overrides the defaults.

    Returns:
        an :term:`Engine`.
    """
    settings = settings or {}
    initial_state = initial_state or {}

    override_topology = settings.get('topology', {})
    processes = {'process': process}
    topology = {
        'process': {
            port: override_topology.get(port, (port,))
            for port in process.ports_schema().keys()}}

    composite = Composite({
        'processes': processes,
        'topology': topology})

    return composite_in_experiment(
        composite=composite,
        settings=settings,
        initial_state=initial_state
    )


def composite_in_experiment(
        composite: Composite,
        settings: Dict[str, Any] = None,
        initial_state: Dict[str, Any] = None,
) -> Engine:
    """Put a Composite in an Engine

    Args:
        composite: the :term:`Composite` object.
        settings: a dictionary of options, including composite_config
            for configuring the composite. Additional  keywords include
            timeline, environment, and outer_path.
        initial_state: initial state to overrides the defaults.

    Returns:
        an :term:`Engine`.
    """
    settings = settings or {}
    initial_state = initial_state or {}

    processes = composite['processes']
    topology = composite['topology']

    timeline = settings.get('timeline', None)
    if timeline is not None:
        add_timeline(processes, topology, timeline)
        all_times = [t[0] for t in timeline['timeline']]
        settings['total_time'] = max(all_times)

    environment = settings.get('environment', None)
    if environment is not None:
        add_environment(processes, topology, environment)

    # initialize the experiment
    experiment_config = {}
    for key, setting in settings.items():
        if key in experiment_config_keys:
            experiment_config[key] = setting
    return Engine(
        composite=composite,
        initial_state=initial_state,
        **experiment_config)


def composer_in_experiment(
        composer: Composer,
        settings: Dict[str, Any] = None,
        initial_state: Dict[str, Any] = None,
        config: Dict[str, Any] = None,
        outer_path: HierarchyPath = (),
) -> Engine:
    """Generate a Composite in an Engine

    Args:
        composer: a :term:`Composer` object.
        settings: a dictionary of options, including composite_config
            for configuring the composite. Additional  keywords include
            timeline, environment, and outer_path.
        initial_state: initial state to overrides the defaults.
        config: updates values in composer's config
        outer_path: path to the processes and topology

    Returns:
        an :term:`Engine`
    """
    composite = composer.generate(config, outer_path)
    return composite_in_experiment(
        composite=composite,
        settings=settings,
        initial_state=initial_state,
    )


# simulate helper functions

def simulate_process(
        process: Process,
        settings: Optional[Dict[str, Any]] = None
) -> Dict:
    """Put a :term:`Process` in an :term:`Engine` and simulate it"""
    settings = settings or {}
    experiment = process_in_experiment(process, settings)
    return simulate_experiment(experiment, settings)


def simulate_composite(
        composite: Composite,
        settings: Optional[Dict[str, Any]] = None
) -> Dict:
    """Put a :term:`Composite` in an :term:`Engine` and simulate it"""
    settings = settings or {}
    experiment = composite_in_experiment(composite, settings)
    return simulate_experiment(experiment, settings)


def simulate_composer(
        composer: Composer,
        settings: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Initialize a :term:`Composer` in an :term:`Engine` and simulate it"""
    settings = settings or {}
    config = config or {}
    outer_path = settings.get('outer_path', tuple())
    composite = composer.generate(config, outer_path)
    initial_state = settings.get('initial_state')
    experiment = composite_in_experiment(
        composite,
        settings=settings,
        initial_state=initial_state)
    return simulate_experiment(experiment, settings)


def simulate_experiment(
        experiment: Engine,
        settings: Optional[Dict[str, Any]] = None
) -> Dict:
    """Simulate an :term:`Engine`.

    Args:
        experiment: a configured experiment

    Returns:
        A timeseries of variables from all ports. If ``return_raw_data``
        is True, return the raw data instead.
    """
    settings = settings or {}
    total_time = settings.get('total_time', 10)
    return_raw_data = settings.get('return_raw_data', False)

    # run simulation
    experiment.update(total_time)
    experiment.end()

    # return data from emitter
    if return_raw_data:
        return experiment.emitter.get_data()
    return experiment.emitter.get_timeseries()


# Tests
def test_process_in_experiment() -> None:
    process = ExchangeA()
    experiment = process_in_experiment(process)
    assert experiment.processes['process'] is process


def test_process_in_experiment_timeline() -> None:
    timeline = [
        (0, {('internal', 'A'): 0}),
        (1, {('internal', 'A'): 1}),
    ]
    process = ExchangeA()
    experiment = process_in_experiment(
        process,
        settings={
            'timeline': {'timeline': timeline}})
    assert experiment.processes['process'] is process
    assert isinstance(
        experiment.processes['timeline'],
        TimelineProcess)


def test_process_in_experiment_environment() -> None:
    process = ExchangeA()
    experiment = process_in_experiment(
        process,
        settings={'environment': {}})

    assert experiment.processes['process'] is process
    assert isinstance(
        experiment.processes['nonspatial_environment'],
        NonSpatialEnvironment)


def test_composer_in_experiment() -> None:
    composer = PoQo({
        '_schema': {'po': {'A': {'a1': {'_emit': True}}}}})

    timeline = [
        (0, {('aaa', 'a1'): 50}),
        (5, {('aaa', 'a1'): 10}),
        (15, {})]
    settings = {
        'environment': {},
        'timeline': {'timeline': timeline}}
    experiment = composer_in_experiment(composer=composer, settings=settings)

    assert isinstance(
        experiment.processes['nonspatial_environment'],
        NonSpatialEnvironment)

    output = simulate_experiment(experiment, settings)

    # check that timeline worked
    assert output['aaa']['a1'][6] == 10
    assert settings['total_time'] == 15
    assert len(output['aaa']['a1']) == 16


def test_composite_in_experiment() -> None:
    composer = PoQo({
        '_schema': {'po': {'A': {'a2': {'_emit': True}}}}})
    composite = composer.generate()
    settings: Dict = {}
    experiment = composite_in_experiment(
        composite=composite,
        settings=settings)
    assert experiment.processes['po'] is composite['processes']['po']

    output = simulate_composite(composite, settings)
    assert output['aaa']['x'][-1] == -90


def test_process_deletion() -> None:
    """Check that processes are successfully deleted"""
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


def test_composer() -> Dict:
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
    return simulate_composer(toy_compartment, settings)


if __name__ == '__main__':
    # test_process_deletion()
    # test_composer()
    # test_replace_names()
    # test_process_in_experiment_environment()
    # test_composer_in_experiment()

    test_composite_in_experiment()  # pragma: no cover
