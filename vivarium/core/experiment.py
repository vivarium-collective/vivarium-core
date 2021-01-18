'''
==========================================
Experiment and Store Classes
==========================================
'''

import os
import logging as log
import pprint
from typing import Any, Dict
import math
import datetime
import time as clock
import uuid

from pymongo.errors import PyMongoError

from vivarium.core.tree import hierarchy_depth, Store
from vivarium.core.emitter import get_emitter
from vivarium.core.process import (
    HierarchyPath,
    Deriver,
    Process,
    ParallelProcess,
    serialize_dictionary,
)
from vivarium.library.topology import (
    get_in, delete_in, assoc_path,
    inverse_topology
)

pretty = pprint.PrettyPrinter(indent=2)


def pp(x):
    pretty.pprint(x)


def pf(x):
    return pretty.pformat(x)


log.basicConfig(level=os.environ.get("LOGLEVEL", log.WARNING))


def starts_with(a_list, sub):
    return len(sub) <= len(a_list) and all(
        a_list[i] == el
        for i, el in enumerate(sub))


def invert_topology(update, args):
    path, topology = args
    return inverse_topology(path[:-1], update, topology)


def generate_state(processes, topology, initial_state):
    state = Store({})
    state.generate_paths(processes, topology)
    state.apply_subschemas()
    state.set_value(initial_state)
    state.apply_defaults()

    return state


def timestamp(dt=None):
    if not dt:
        dt = datetime.datetime.now()
    return "%04d%02d%02d.%02d%02d%02d" % (
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second)


def invoke_process(process, interval, states):
    return process.next_update(interval, states)


class Defer:
    def __init__(self, defer, f, args):
        self.defer = defer
        self.f = f
        self.args = args

    def get(self):
        return self.f(
            self.defer.get(),
            self.args)


class InvokeProcess:
    def __init__(self, process, interval, states):
        self.process = process
        self.interval = interval
        self.states = states
        self.update = invoke_process(
            self.process,
            self.interval,
            self.states)

    def get(self):
        return self.update


class MultiInvoke:
    def __init__(self, pool):
        self.pool = pool

    def invoke(self, process, interval, states):
        args = (process, interval, states)
        result = self.pool.apply_async(invoke_process, args)
        return result


class Experiment:
    def __init__(self, config):
        # type: (Dict[str, Any]) -> None
        """Defines simulations

        Arguments:
            config (dict): A dictionary of configuration options. The
                required options are:

                * **processes** (:py:class:`dict`): A dictionary that
                    maps :term:`process` names to process objects. You
                    will usually get this from the ``processes``
                    attribute of the dictionary from
                    :py:meth:`vivarium.core.process.Factory.generate`.
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
        self.emit_step = config.get('emit_step')

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
        self.deriver_paths: Dict[HierarchyPath, Deriver] = {}
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
        emitter_config['experiment_id'] = self.experiment_id
        self.emitter = get_emitter(emitter_config)

        self.local_time = 0.0

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

    def add_process_path(self, process, path):
        if process.is_deriver():
            self.deriver_paths[path] = process
        else:
            self.process_paths[path] = process

    def find_process_paths(self, processes):
        tree = hierarchy_depth(processes)
        for path, process in tree.items():
            self.add_process_path(process, path)

    def emit_configuration(self):
        data = {
            'time_created': self.time_created,
            'experiment_id': self.experiment_id,
            'name': self.experiment_name,
            'description': self.description,
            'topology': self.topology,
            'processes': serialize_dictionary(self.processes),
            'state': self.state.get_config()
        }
        emit_config = {
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

    def invoke_process(self, process, path, interval, states):
        if process.parallel:
            # add parallel process if it doesn't exist
            if path not in self.parallel:
                self.parallel[path] = ParallelProcess(process)
            # trigger the computation of the parallel process
            self.parallel[path].update(interval, states)

            return self.parallel[path]
        # if not parallel, perform a normal invocation
        return self.invoke(process, interval, states)

    def process_update(self, path, process, interval):
        state = self.state.get_path(path)
        process_topology = get_in(self.topology, path)

        # translate the values from the tree structure into the form
        # that this process expects, based on its declared topology
        states = state.outer.schema_topology(process.schema, process_topology)

        update = self.invoke_process(
            process,
            path,
            interval,
            states)

        absolute = Defer(update, invert_topology, (path, process_topology))

        return absolute, state

    def apply_update(self, update, state):
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

    def delete_path(self, deletion):
        delete_in(self.processes, deletion)
        delete_in(self.topology, deletion)

        for path in list(self.process_paths.keys()):
            if starts_with(path, deletion):
                del self.process_paths[path]

        for path in list(self.deriver_paths.keys()):
            if starts_with(path, deletion):
                del self.deriver_paths[path]

    def run_derivers(self):
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
                update, state = self.process_update(
                    path, deriver, 0)
                self.apply_update(update.get(), state)

    def emit_data(self):
        data = self.state.emit_data()
        data.update({
            'time': self.local_time})
        emit_config = {
            'table': 'history',
            'data': serialize_dictionary(data)}
        self.emitter.emit(emit_config)

    def send_updates(self, update_tuples):
        for update_tuple in update_tuples:
            update, state = update_tuple
            self.apply_update(update.get(), state)

        self.run_derivers()

    def update(self, interval):
        """ Run each process for the given interval and update the states.
        """

        time = 0
        emit_time = self.emit_step
        clock_start = clock.time()

        def empty_front(t):
            return {
                'time': t,
                'update': {}}

        # keep track of which processes have simulated until when
        front = {}

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
                    future = min(
                        process_time + process.local_timestep(),
                        interval)
                    timestep = future - process_time

                    # calculate the update for this process
                    # TODO(jerry): Do something cleaner than having
                    #  generate_paths() add a schema attribute to the Process.
                    #  PyCharm's type check reports:
                    #    Type Process doesn't have expected attribute 'schema'
                    # TODO(chris): Is there any reason to generate a process's
                    #  schema dynamically like this?
                    update = self.process_update(path, process, timestep)

                    # store the update to apply at its projected time
                    if timestep < full_step:
                        full_step = timestep
                    front[path]['time'] = future
                    front[path]['update'] = update

            if full_step == math.inf:
                # no processes ran, jump to next process
                next_event = interval
                for path in front.keys():
                    if front[path]['time'] < next_event:
                        next_event = front[path]['time']
                time = next_event
            else:
                # at least one process ran, apply updates and continue
                future = time + full_step

                updates = []
                paths = []
                for path, advance in front.items():
                    if advance['time'] <= future:
                        new_update = advance['update']
                        # new_update['_path'] = path
                        updates.append(new_update)
                        advance['update'] = {}
                        paths.append(path)

                self.send_updates(updates)

                time = future
                self.local_time += full_step

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

    def end(self):
        for parallel in self.parallel.values():
            parallel.end()

    def print_display(self):
        date, time = self.time_created.split('.')
        print('\nExperiment ID: {}'.format(self.experiment_id))
        print('Created: {} at {}'.format(
            date[4:6] + '/' + date[6:8] + '/' + date[0:4],
            time[0:2] + ':' + time[2:4] + ':' + time[4:6]))
        if self.experiment_name is not self.experiment_id:
            print('Name: {}'.format(self.experiment_name))
        if self.description:
            print('Description: {}'.format(self.description))

    def print_summary(self, clock_finish):
        if clock_finish < 1:
            print('Completed in {:.6f} seconds'.format(clock_finish))
        else:
            print('Completed in {:.2f} seconds'.format(clock_finish))


def print_progress_bar(
        iteration,
        total,
        decimals=1,
        length=50,
):
    """ Call in a loop to create terminal progress bar

    Arguments:
        iteration: (Required) current iteration
        total:     (Required) total iterations
        decimals:  (Optional) positive number of decimals in percent complete
        length:    (Optional) character length of bar
    """
    progress = ("{0:." + str(decimals) + "f}").format(total - iteration)
    filled_length = int(length * iteration // total)
    filled_bar = '█' * filled_length + '-' * (length - filled_length)
    print(
        f'\rProgress:|{filled_bar}| {progress}/{float(total)} '
        f'simulated seconds remaining    ', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
