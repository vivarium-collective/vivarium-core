"""
==========
Experiment
==========

Experiment runs the simulation.
"""

import os
import logging as log
import pprint
import multiprocessing
from multiprocessing import pool as multipool
from typing import (
    Any, Dict, Optional, Union, Tuple, Callable)
import math
import datetime
import time as clock
import uuid

from pymongo.errors import PyMongoError

from vivarium.composites.toys import Proton, Electron, Sine, PoQo
from vivarium.core.store import hierarchy_depth, Store, generate_state
from vivarium.core.emitter import get_emitter
from vivarium.core.process import (
    Process,
    ParallelProcess,
    serialize_value,
    Composer,
)
from vivarium.library.topology import (
    delete_in,
    assoc_path,
    inverse_topology
)
from vivarium.library.units import units
from vivarium.core.types import (
    HierarchyPath, Topology, Schema, State, Update, Processes)

pretty = pprint.PrettyPrinter(indent=2)


def pp(x: Any) -> None:
    pretty.pprint(x)


def pf(x: Any) -> str:
    return pretty.pformat(x)


log.basicConfig(level=os.environ.get("LOGLEVEL", log.WARNING))


def starts_with(
        a_list: HierarchyPath,
        sub: HierarchyPath,
) -> bool:
    return len(sub) <= len(a_list) and all(
        a_list[i] == el
        for i, el in enumerate(sub))


def invert_topology(
        update: Update,
        args: Tuple[HierarchyPath, Topology],
) -> State:
    path, topology = args
    return inverse_topology(path[:-1], update, topology)


def timestamp(dt: Optional[Any] = None) -> str:
    if not dt:
        dt = datetime.datetime.now()
    return "%04d%02d%02d.%02d%02d%02d" % (
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second)


def invoke_process(
        process: Process,
        interval: float,
        states: State,
) -> Update:
    return process.next_update(interval, states)


class Defer:
    def __init__(
            self,
            defer: Any,
            f: Callable,
            args: Tuple,
    ) -> None:
        self.defer = defer
        self.f = f
        self.args = args

    def get(self) -> Update:
        return self.f(
            self.defer.get(),
            self.args)


class InvokeProcess:
    def __init__(
            self,
            process: Process,
            interval: float,
            states: State,
    ) -> None:
        self.process = process
        self.interval = interval
        self.states = states
        self.update = invoke_process(
            self.process,
            self.interval,
            self.states)

    def get(self) -> Update:
        return self.update


class MultiInvoke:
    def __init__(
            self,
            pool: multipool.Pool
    ) -> None:
        self.pool = pool

    def invoke(
            self,
            process: Process,
            interval: float,
            states: State,
    ) -> Any:
        args = (process, interval, states)
        result = self.pool.apply_async(invoke_process, args)
        return result


class Experiment:
    def __init__(self, config: Dict[str, Any]) -> None:
        """Defines simulations

        Arguments:
            config (dict): A dictionary of configuration options. The
                required options are:

                * **processes** (:py:class:`dict`): A dictionary that
                    maps :term:`process` names to process objects. You
                    will usually get this from the ``processes``
                    attribute of the dictionary from
                    :py:meth:`vivarium.core.process.Composer.generate`.
                * **topology** (:py:class:`dict`): A dictionary that
                    maps process names to sub-dictionaries. These
                    sub-dictionaries map the process's port names to
                    tuples that specify a path through the :term:`tree`
                    from the :term:`compartment` root to the
                    :term:`store` that will be passed to the process for
                    that port.

                The following options are optional:

                * **experiment_id** (:py:class:`uuid.UUID` or
                    :py:class:`str`): A unique identifier for the
                    experiment. A UUID will be generated if none is
                    provided.
                * **description** (:py:class:`str`): A description of
                    the experiment. A blank string by default.
                * **initial_state** (:py:class:`dict`): By default an
                    empty dictionary, this is the initial state of the
                    simulation.
                * **emitter** (:py:class:`dict`): An emitter
                    configuration which must conform to the
                    specification in the documentation for
                    :py:func:`vivarium.core.emitter.get_emitter`. The
                    experiment ID will be added to the dictionary you
                    provide as the value for the key ``experiment_id``.
        """
        self.config = config
        self.experiment_id = config.get(
            'experiment_id', str(uuid.uuid1()))
        self.processes = config['processes']
        self.topology = config['topology']
        self.initial_state = config.get('initial_state', {})
        self.emit_step = config.get('emit_step', 1.0)

        # display settings
        self.experiment_name = config.get(
            'experiment_name', self.experiment_id)
        self.description = config.get('description', '')
        self.display_info = config.get('display_info', True)
        self.progress_bar = config.get('progress_bar', False)
        self.time_created = timestamp()
        if self.display_info:
            self.print_display()

        # parallel settings
        self.invoke = config.get('invoke', InvokeProcess)
        self.parallel: Dict[HierarchyPath, ParallelProcess] = {}

        # get a mapping of all paths to processes
        self.process_paths: Dict[HierarchyPath, Process] = {}
        self.deriver_paths: Dict[HierarchyPath, Process] = {}
        self.find_process_paths(self.processes)

        # initialize the state
        self.state = generate_state(
            self.processes,
            self.topology,
            self.initial_state)

        # emitter settings
        emitter_config = config.get('emitter', 'timeseries')
        if isinstance(emitter_config, str):
            emitter_config = {'type': emitter_config}
        else:
            emitter_config = dict(emitter_config)
        emitter_config['experiment_id'] = self.experiment_id
        self.emitter = get_emitter(emitter_config)

        self.experiment_time = 0.0

        # run the derivers
        self.send_updates([])

        # run the emitter
        self.emit_configuration()
        self.emit_data()

        # logging information
        log.info('experiment %s', str(self.experiment_id))

        log.info('\nPROCESSES:')
        log.info(pf(self.processes))

        log.info('\nTOPOLOGY:')
        log.info(pf(self.topology))

        # log.info('\nSTATE:')
        # log.info(pf(self.state.get_value()))
        #
        # log.info('\nCONFIG:')
        # log.info(pf(self.state.get_config(True)))

    def add_process_path(
            self,
            process: Process,
            path: HierarchyPath
    ) -> None:
        if process.is_deriver():
            self.deriver_paths[path] = process
        else:
            self.process_paths[path] = process

    def find_process_paths(
            self,
            processes: Processes
    ) -> None:
        tree = hierarchy_depth(processes)
        for path, process in tree.items():
            self.add_process_path(process, path)

    def emit_configuration(self) -> None:
        data: Dict[str, Any] = {
            'time_created': self.time_created,
            'experiment_id': self.experiment_id,
            'name': self.experiment_name,
            'description': self.description,
            'topology': self.topology,
            'processes': serialize_value(self.processes),
            'state': serialize_value(self.state.get_config())
        }
        emit_config: Dict[str, Any] = {
            'table': 'configuration',
            'data': data}
        try:
            self.emitter.emit(emit_config)
        except PyMongoError:
            log.exception("emitter.emit", exc_info=True, stack_info=True)
            # TODO -- handle large parameter sets to meet mongoDB limit
            del emit_config['data']['processes']
            del emit_config['data']['state']
            self.emitter.emit(emit_config)

    def invoke_process(
            self,
            process: Process,
            path: HierarchyPath,
            interval: float,
            states: State,
    ) -> Any:
        if process.parallel:
            # add parallel process if it doesn't exist
            if path not in self.parallel:
                self.parallel[path] = ParallelProcess(process)
            # trigger the computation of the parallel process
            self.parallel[path].update(interval, states)

            return self.parallel[path]
        # if not parallel, perform a normal invocation
        return self.invoke(process, interval, states)

    def process_update(
            self,
            path: HierarchyPath,
            process: Process,
            store: Store,
            states: State,
            interval: float,
    ) -> Tuple[Defer, Store]:

        update = self.invoke_process(
            process,
            path,
            interval,
            states)

        absolute = Defer(
            update,
            invert_topology,
            (path, store.topology))

        return absolute, store

    def process_state(
            self,
            path: HierarchyPath,
            process: Process,
    ) -> Tuple[Store, State]:
        store = self.state.get_path(path)

        # translate the values from the tree structure into the form
        # that this process expects, based on its declared topology
        states = store.outer.schema_topology(process.schema, store.topology)

        return store, states

    def calculate_update(
            self,
            path: HierarchyPath,
            process: Process,
            interval: float
    ) -> Tuple[Defer, Store]:
        store, states = self.process_state(path, process)
        return self.process_update(
            path, process, store, states, interval)

    def apply_update(
            self,
            update: Update,
            state: Store
    ) -> None:
        topology_updates, process_updates, deletions = self.state.apply_update(
            update, state)

        if topology_updates:
            for path, topology_update in topology_updates:
                assoc_path(self.topology, path, topology_update)

        if process_updates:
            for path, process in process_updates:
                assoc_path(self.processes, path, process)
                self.add_process_path(process, path)

        if deletions:
            for deletion in deletions:
                self.delete_path(deletion)

    def delete_path(
            self,
            deletion: HierarchyPath
    ) -> None:
        delete_in(self.processes, deletion)
        delete_in(self.topology, deletion)

        for path in list(self.process_paths.keys()):
            if starts_with(path, deletion):
                del self.process_paths[path]

        for path in list(self.deriver_paths.keys()):
            if starts_with(path, deletion):
                del self.deriver_paths[path]

    def run_derivers(self) -> None:
        paths = list(self.deriver_paths.keys())
        for path in paths:
            # deriver could have been deleted by another deriver
            deriver = self.deriver_paths.get(path)
            if deriver:
                # timestep shouldn't influence derivers
                # TODO(jerry): Do something cleaner than having
                #  generate_paths() add a schema attribute to the Deriver.
                #  PyCharm's type check reports:
                #    Type Process doesn't have expected attribute 'schema'
                update, store = self.calculate_update(
                    path, deriver, 0)
                self.apply_update(update.get(), store)

    def emit_data(self) -> None:
        data = self.state.emit_data()
        data.update({
            'time': self.experiment_time})
        emit_config = {
            'table': 'history',
            'data': serialize_value(data)}
        self.emitter.emit(emit_config)

    def send_updates(
            self,
            update_tuples: list
    ) -> None:
        for update_tuple in update_tuples:
            update, state = update_tuple
            self.apply_update(update.get(), state)

        self.run_derivers()

    def update(
            self,
            interval: float
    ) -> None:
        """ Run each process for the given interval and update the states.
        """

        time = 0.0
        emit_time = self.emit_step
        clock_start = clock.time()

        def empty_front(t: float) -> Dict[str, Union[float, dict]]:
            return {
                'time': t,
                'update': {}}

        # keep track of which processes have simulated until when
        front: Dict = {}

        while time < interval:
            full_step = math.inf

            # find any parallel processes that were removed and terminate them
            for terminated in self.parallel.keys() - self.process_paths.keys():
                self.parallel[terminated].end()
                del self.parallel[terminated]

            # setup a way to track how far each process has simulated in time
            front = {
                path: progress
                for path, progress in front.items()
                if path in self.process_paths}

            # go through each process and find those that are able to update
            # based on their current time being less than the global time.
            for path, process in self.process_paths.items():
                if path not in front:
                    front[path] = empty_front(time)
                process_time = front[path]['time']

                if process_time <= time:

                    # get the time step
                    store, states = self.process_state(path, process)
                    requested_timestep = process.calculate_timestep(states)

                    # progress only to the end of interval
                    future = min(process_time + requested_timestep, interval)
                    process_timestep = future - process_time

                    # calculate the update for this process
                    # TODO(jerry): Do something cleaner than having
                    #  generate_paths() add a schema attribute to the Process.
                    #  PyCharm's type check reports:
                    #    Type Process doesn't have expected attribute 'schema'
                    # TODO(chris): Is there any reason to generate a process's
                    #  schema dynamically like this?
                    update = self.process_update(
                        path, process, store, states, process_timestep)

                    # store the update to apply at its projected time
                    front[path]['time'] = future
                    front[path]['update'] = update

                    # absolute timestep
                    timestep = future - time
                    if timestep < full_step:
                        full_step = timestep

                else:
                    # don't shoot past processes that didn't run this time
                    process_delay = process_time - time
                    if process_delay < full_step:
                        full_step = process_delay

            if full_step == math.inf:
                # no processes ran, jump to next process
                next_event = interval
                for path in front.keys():
                    if front[path]['time'] < next_event:
                        next_event = front[path]['time']
                time = next_event
            else:
                # at least one process ran
                # increase the time, apply updates, and continue
                time += full_step
                self.experiment_time += full_step

                updates = []
                paths = []
                for path, advance in front.items():
                    if advance['time'] <= time:
                        new_update = advance['update']
                        # new_update['_path'] = path
                        updates.append(new_update)
                        advance['update'] = {}
                        paths.append(path)

                self.send_updates(updates)

                # display and emit
                if self.progress_bar:
                    print_progress_bar(time, interval)
                if self.emit_step is None:
                    self.emit_data()
                elif emit_time <= time:
                    while emit_time <= time:
                        self.emit_data()
                        emit_time += self.emit_step

        # post-simulation
        for advance in front.values():
            assert advance['time'] == time == interval
            assert len(advance['update']) == 0

        clock_finish = clock.time() - clock_start

        if self.display_info:
            self.print_summary(clock_finish)

    def end(self) -> None:
        for parallel in self.parallel.values():
            parallel.end()

    def print_display(self) -> None:
        date, time = self.time_created.split('.')
        print('\nExperiment ID: {}'.format(self.experiment_id))
        print('Created: {} at {}'.format(
            date[4:6] + '/' + date[6:8] + '/' + date[0:4],
            time[0:2] + ':' + time[2:4] + ':' + time[4:6]))
        if self.experiment_name is not self.experiment_id:
            print('Name: {}'.format(self.experiment_name))
        if self.description:
            print('Description: {}'.format(self.description))

    def print_summary(
            self,
            clock_finish: float
    ) -> None:
        if clock_finish < 1:
            print('Completed in {:.6f} seconds'.format(clock_finish))
        else:
            print('Completed in {:.2f} seconds'.format(clock_finish))


def print_progress_bar(
        iteration: float,
        total: float,
        decimals: float = 1,
        length: int = 50,
) -> None:
    """ Call in a loop to create terminal progress bar

    Arguments:
        iteration: (Required) current iteration
        total:     (Required) total iterations
        decimals:  (Optional) positive number of decimals in percent complete
        length:    (Optional) character length of bar
    """
    progress: str = ("{0:." + str(decimals) + "f}").format(total - iteration)
    filled_length: int = int(length * iteration // total)
    filled_bar = '█' * filled_length + '-' * (length - filled_length)
    print(
        f'\rProgress:|{filled_bar}| {progress}/{float(total)} '
        f'simulated seconds remaining    ', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def make_proton(
        parallel: bool = False
) -> Dict[str, Any]:
    processes = {
        'proton': Proton({'_parallel': parallel}),
        'electrons': {
            'a': {
                'electron': Electron({'_parallel': parallel})},
            'b': {
                'electron': Electron()}}}

    spin_path = ('internal', 'spin')
    radius_path = ('structure', 'radius')

    topology = {
        'proton': {
            'radius': radius_path,
            'quarks': ('internal', 'quarks'),
            'electrons': {
                '_path': ('electrons',),
                '*': {
                    'orbital': ('shell', 'orbital'),
                    'spin': spin_path}}},
        'electrons': {
            'a': {
                'electron': {
                    'spin': spin_path,
                    'proton': {
                        '_path': ('..', '..'),
                        'radius': radius_path}}},
            'b': {
                'electron': {
                    'spin': spin_path,
                    'proton': {
                        '_path': ('..', '..'),
                        'radius': radius_path}}}}}

    initial_state = {
        'structure': {
            'radius': 0.7},
        'internal': {
            'quarks': {
                'x': {
                    'color': 'green',
                    'spin': 'up'},
                'y': {
                    'color': 'red',
                    'spin': 'up'},
                'z': {
                    'color': 'blue',
                    'spin': 'down'}}}}

    return {
        'processes': processes,
        'topology': topology,
        'initial_state': initial_state}


def test_recursive_store() -> None:
    environment_config = {
        'environment': {
            'temperature': {
                '_default': 0.0,
                '_updater': 'accumulate'},
            'fields': {
                (0, 1): {
                    'enzymeX': {
                        '_default': 0.0,
                        '_updater': 'set'},
                    'enzymeY': {
                        '_default': 0.0,
                        '_updater': 'set'}},
                (0, 2): {
                    'enzymeX': {
                        '_default': 0.0,
                        '_updater': 'set'},
                    'enzymeY': {
                        '_default': 0.0,
                        '_updater': 'set'}}},
            'agents': {
                '1': {
                    'location': {
                        '_default': (0, 0),
                        '_updater': 'set'},
                    'boundary': {
                        'external': {
                            '_default': 0.0,
                            '_updater': 'set'},
                        'internal': {
                            '_default': 0.0,
                            '_updater': 'set'}},
                    'transcripts': {
                        'flhDC': {
                            '_default': 0,
                            '_updater': 'accumulate'},
                        'fliA': {
                            '_default': 0,
                            '_updater': 'accumulate'}},
                    'proteins': {
                        'ribosome': {
                            '_default': 0,
                            '_updater': 'set'},
                        'flagella': {
                            '_default': 0,
                            '_updater': 'accumulate'}}},
                '2': {
                    'location': {
                        '_default': (0, 0),
                        '_updater': 'set'},
                    'boundary': {
                        'external': {
                            '_default': 0.0,
                            '_updater': 'set'},
                        'internal': {
                            '_default': 0.0,
                            '_updater': 'set'}},
                    'transcripts': {
                        'flhDC': {
                            '_default': 0,
                            '_updater': 'accumulate'},
                        'fliA': {
                            '_default': 0,
                            '_updater': 'accumulate'}},
                    'proteins': {
                        'ribosome': {
                            '_default': 0,
                            '_updater': 'set'},
                        'flagella': {
                            '_default': 0,
                            '_updater': 'accumulate'}}}}}}

    state = Store(environment_config)
    state.apply_update({})
    state.state_for(['environment'], ['temperature'])


def test_topology_ports() -> None:
    proton = make_proton()

    experiment = Experiment(proton)

    log.debug(pf(experiment.state.get_config(True)))

    experiment.update(10.0)

    log.debug(pf(experiment.state.get_config(True)))
    log.debug(pf(experiment.state.divide_value()))


def test_timescales() -> None:
    class Slow(Process):
        name = 'slow'
        defaults = {'timestep': 3.0}

        def __init__(self, config: Optional[dict] = None) -> None:
            super().__init__(config)

        def ports_schema(self) -> Schema:
            return {
                'state': {
                    'base': {
                        '_default': 1.0}}}

        def next_update(
                self,
                timestep: Union[float, int],
                states: State) -> Update:
            base = states['state']['base']
            next_base = timestep * base * 0.1

            return {
                'state': {'base': next_base}}

    class Fast(Process):
        name = 'fast'
        defaults = {'timestep': 0.3}

        def __init__(self, config: Optional[dict] = None) -> None:
            super().__init__(config)

        def ports_schema(self) -> Schema:
            return {
                'state': {
                    'base': {
                        '_default': 1.0},
                    'motion': {
                        '_default': 0.0}}}

        def next_update(
                self,
                timestep: Union[float, int],
                states: State) -> Update:
            base = states['state']['base']
            motion = timestep * base * 0.001

            return {
                'state': {'motion': motion}}

    processes = {
        'slow': Slow(),
        'fast': Fast()}

    states = {
        'state': {
            'base': 1.0,
            'motion': 0.0}}

    topology = {
        'slow': {'state': ('state',)},
        'fast': {'state': ('state',)}}

    emitter = {'type': 'null'}
    experiment = Experiment({
        'processes': processes,
        'topology': topology,
        'emitter': emitter,
        'initial_state': states})

    experiment.update(10.0)


def test_2_store_1_port() -> None:
    """
    Split one port of a processes into two stores
    """
    class OnePort(Process):
        name = 'one_port'

        def ports_schema(self) -> Schema:
            return {
                'A': {
                    'a': {
                        '_default': 0,
                        '_emit': True},
                    'b': {
                        '_default': 0,
                        '_emit': True}
                }
            }

        def next_update(
                self,
                timestep: Union[float, int],
                states: State) -> Update:
            return {
                'A': {
                    'a': 1,
                    'b': 2}}

    class SplitPort(Composer):
        """splits OnePort's ports into two stores"""
        name = 'split_port_composer'

        def generate_processes(
                self, config: Optional[dict]) -> Dict[str, Any]:
            return {
                'one_port': OnePort({})}

        def generate_topology(self, config: Optional[dict]) -> Topology:
            return {
                'one_port': {
                    'A': {
                        'a': ('internal', 'a',),
                        'b': ('external', 'a',)
                    }
                }}

    # run experiment
    split_port = SplitPort({})
    network = split_port.generate()
    exp = Experiment({
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'external': {'a': [0, 2, 4]},
        'internal': {'a': [0, 1, 2]},
        'time': [0.0, 1.0, 2.0]}
    assert output == expected_output


def test_multi_port_merge() -> None:
    class MultiPort(Process):
        name = 'multi_port'

        def ports_schema(self) -> Schema:
            return {
                'A': {
                    'a': {
                        '_default': 0,
                        '_emit': True}},
                'B': {
                    'a': {
                        '_default': 0,
                        '_emit': True}},
                'C': {
                    'a': {
                        '_default': 0,
                        '_emit': True}}}

        def next_update(
                self,
                timestep: Union[float, int],
                states: State) -> Update:
            return {
                'A': {'a': 1},
                'B': {'a': 1},
                'C': {'a': 1}}

    class MergePort(Composer):
        """combines both of MultiPort's ports into one store"""
        name = 'multi_port_composer'

        def generate_processes(
                self, config: Optional[dict]) -> Dict[str, Any]:
            return {
                'multi_port': MultiPort({})}

        def generate_topology(self, config: Optional[dict]) -> Topology:
            return {
                'multi_port': {
                    'A': ('aaa',),
                    'B': ('aaa',),
                    'C': ('aaa',)}}

    # run experiment
    merge_port = MergePort({})
    network = merge_port.generate()
    exp = Experiment({
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'aaa': {'a': [0, 3, 6]},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


def test_complex_topology() -> None:

    # make the experiment
    outer_path = ('universe', 'agent')
    pq = PoQo({})
    pq_composite = pq.generate(path=outer_path)
    experiment = Experiment(pq_composite)

    # get the initial state
    initial_state = experiment.state.get_value()
    print('time 0:')
    pp(initial_state)

    # simulate for 1 second
    experiment.update(1)

    next_state = experiment.state.get_value()
    print('time 1:')
    pp(next_state)

    # pull out the agent state
    initial_agent_state = initial_state['universe']['agent']
    agent_state = next_state['universe']['agent']

    assert agent_state['aaa']['a1'] == initial_agent_state['aaa']['a1'] + 1
    assert agent_state['aaa']['x'] == initial_agent_state['aaa']['x'] - 9
    assert agent_state['ccc']['a3'] == initial_agent_state['ccc']['a3'] + 1


def test_multi() -> None:
    with multiprocessing.Pool(processes=4) as pool:
        multi = MultiInvoke(pool)
        proton = make_proton()
        experiment = Experiment({**proton, 'invoke': multi.invoke})

        log.debug(pf(experiment.state.get_config(True)))

        experiment.update(10.0)

        log.debug(pf(experiment.state.get_config(True)))
        log.debug(pf(experiment.state.divide_value()))


def test_parallel() -> None:
    proton = make_proton(parallel=True)
    experiment = Experiment(proton)

    log.debug(pf(experiment.state.get_config(True)))

    experiment.update(10.0)

    log.debug(pf(experiment.state.get_config(True)))
    log.debug(pf(experiment.state.divide_value()))

    experiment.end()


def test_depth() -> None:
    nested = {
        'A': {
            'AA': 5,
            'AB': {
                'ABC': 11}},
        'B': {
            'BA': 6}}

    dissected = hierarchy_depth(nested)
    assert len(dissected) == 3
    assert dissected[('A', 'AB', 'ABC')] == 11


def test_sine() -> None:
    sine = Sine()
    print(sine.next_update(0.25 / 440.0, {
        'frequency': 440.0,
        'amplitude': 0.1,
        'phase': 1.5}))


def test_units() -> None:
    class UnitsMicrometer(Process):
        name = 'units_micrometer'

        def ports_schema(self) -> Schema:
            return {
                'A': {
                    'a': {
                        '_default': 0 * units.um,
                        '_emit': True},
                    'b': {
                        '_default': 'string b',
                        '_emit': True,
                    }
                }
            }

        def next_update(
                self, timestep: Union[float, int], states: State) -> Update:
            return {
                'A': {'a': 1 * units.um}}

    class UnitsMillimeter(Process):
        name = 'units_millimeter'

        def ports_schema(self) -> Schema:
            return {
                'A': {
                    'a': {
                        # '_default': 0 * units.mm,
                        '_emit': True}}}

        def next_update(
                self, timestep: Union[float, int], states: State) -> Update:
            return {
                'A': {'a': 1 * units.mm}}

    class MultiUnits(Composer):
        name = 'multi_units_composer'

        def generate_processes(
                self,
                config: Optional[dict]) -> Dict[str, Any]:
            return {
                'units_micrometer':
                    UnitsMicrometer({}),
                'units_millimeter':
                    UnitsMillimeter({})}

        def generate_topology(self, config: Optional[dict]) -> Topology:
            return {
                'units_micrometer': {
                    'A': ('aaa',)},
                'units_millimeter': {
                    'A': ('aaa',)}}

    # run experiment
    multi_unit = MultiUnits({})
    network = multi_unit.generate()
    exp = Experiment({
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(5)
    timeseries = exp.emitter.get_timeseries()
    print('TIMESERIES')
    pp(timeseries)

    data = exp.emitter.get_data()
    print('DATA')
    pp(data)

    data_deserialized = exp.emitter.get_data_deserialized()
    print('DESERIALIZED')
    pp(data_deserialized)

    data_unitless = exp.emitter.get_data_unitless()
    print('UNITLESS')
    pp(data_unitless)


if __name__ == '__main__':
    # test_recursive_store()
    # test_timescales()
    # test_topology_ports()
    # test_multi()
    # test_sine()
    # test_parallel()
    test_complex_topology()
    # test_multi_port_merge()
    # test_2_store_1_port()
    # test_units()
