"""
Experiment to test maximum BSON document size with MongoDB emitter
"""
import uuid
import random
import time
import cProfile, pstats
from pstats import SortKey
import argparse

from vivarium.core.engine import Engine
from vivarium.core.process import Process
from vivarium.core.composer import Composer
from vivarium.core.emitter import (
    get_experiment_database,
    data_from_database,
    delete_experiment_from_database)


class ManyParametersProcess(Process):
    defaults = {
        'number_of_parameters': 100}
    def __init__(self, parameters=None):
        super().__init__(parameters)
        # make a bunch of parameters
        random_parameters = {
            key: random.random()
            for key in range(self.parameters['number_of_parameters'])}
        super().__init__(random_parameters)
    def ports_schema(self):
        return {'port': {'variable': {'_default': 0, '_emit': True}}}
    def next_update(self, timestep, states):
        return {'port': {'variable': 1}}


class ManyParametersComposite(Composer):
    defaults = {
        'number_of_processes': 10,
        'number_of_parameters': 100}
    def __init__(self, config=None):
        super().__init__(config)
        self.process_ids = [
            f'process_{key}'
            for key in range(self.config['number_of_processes'])]
    def generate_processes(self, config):
        # make a bunch of processes
        return {
            process_id: ManyParametersProcess({
                'name': process_id,
                'number_of_parameters': self.config['number_of_parameters']})
            for process_id in self.process_ids}
    def generate_topology(self, config):
        return {
            process_id: {'port': ('store', process_id,)}
            for process_id in self.process_ids}


class ManyVariablesProcess(Process):
    defaults = {
        'number_of_variables': 10,
        'process_sleep': 0,
    }
    def __init__(self, parameters=None):
        super().__init__(parameters)
        # make a bunch of variables
        random_variables = [key for key in range(self.parameters['number_of_variables'])]
        self.parameters['variables'] = random_variables

    def ports_schema(self):
        return {
            'update_timer': {
                '_default': 0,
                '_emit': True,
            },
            'port': {
                variable: {
                    '_default': random.random(),
                    '_emit': True
                } for variable in self.parameters['variables']}}

    def next_update(self, timestep, states):
        update = {
                variable: random.random()
                for variable in self.parameters['variables']}
        time.sleep(self.parameters['process_sleep'])
        return {
            'port': update,
        }

class ManyVariablesComposite(Composer):
    defaults = {
        'number_of_processes': 10,
        'number_of_stores': 10,
        'variables_per_process': 10,
        'process_sleep': 0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.process_ids = [
            f'process_{key}'
            for key in range(self.config['number_of_processes'])]
        self.store_ids = [
            f'store_{key}'
            for key in range(self.config['number_of_stores'])]

    def generate_processes(self, config):
        # make a bunch of processes
        return {
            process_id: ManyVariablesProcess({
                'name': process_id,
                'number_of_variables': self.config['variables_per_process'],
                'process_sleep': self.config['process_sleep'],
            })
            for process_id in self.process_ids}

    def generate_topology(self, config):
        return {
            process_id: {
                'port': (random.choice(self.store_ids),),
            }
            for process_id in self.process_ids}



def run_large_initial_emit():
    """
    This experiment runs a large experiment to test the database emitter.
    This requires MongoDB to be configured and running.
    """

    config = {
        'number_of_processes': 1000,
        'number_of_parameters': 1000}

    composer = ManyParametersComposite(config)
    composite = composer.generate()

    settings = {
        'experiment_name': 'large database experiment',
        'experiment_id': f'large_{str(uuid.uuid4())}',
        'emitter': 'database',
    }

    experiment = Engine({
        'processes': composite['processes'],
        'topology': composite['topology'],
        **settings})

    # run the experiment
    experiment.update(10)

    # retrieve the data with data_from_database
    experiment_id = experiment.experiment_id

    # retrieve the data from emitter
    data = experiment.emitter.get_data()
    assert list(data.keys()) == [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # retrieve the data directly from database
    db = get_experiment_database()
    data, experiment_config = data_from_database(experiment_id, db)
    assert 'processes' in experiment_config
    assert 0.0 in data

    # delete the experiment
    delete_experiment_from_database(experiment_id)


class ComplexModelSim:
    """Profile Complex Models

    This class lets you initialize and profile the simulation of composite models
    with arbitrary numbers of processes, variables per process, and total stores.
    """

    # model complexity
    number_of_processes = 10
    number_of_variables = 10
    number_of_stores = 10
    process_sleep = 1e-4
    experiment_time = 100

    # scans
    processes_range = [1, 10]
    variables_range = [1, 10]
    stores_range = [1, 10]

    # display
    print_top_stats = 4

    def from_cli(self):
        parser = argparse.ArgumentParser(
            description='complex model simulations with runtime profiling'
        )
        parser.add_argument(
            '--profile', '-p', action="store_true",
            help="run profile of model composition and simulation"
        )
        parser.add_argument(
            '--latency', '-l', action="store_true",
            help="run profile of communication latency in an experiment"
        )
        parser.add_argument(
            '--scan', '-s', action="store_true",
            help="run scan of communication latency"
        )
        args = parser.parse_args()

        if args.profile:
            self.run_profile()
        if args.latency:
            self.profile_communication_latency()
        if args.scan:
            self.run_scan()

    def set_parameters(
            self,
            number_of_processes=None,
            number_of_variables=None,
            number_of_stores=None,
            process_sleep=None,
            print_top_stats=None,
            experiment_time=None,
    ):
        self.number_of_processes = number_of_processes or self.number_of_processes
        self.number_of_variables = number_of_variables or self.number_of_variables
        self.process_sleep = process_sleep or self.process_sleep
        self.print_top_stats = print_top_stats or self.print_top_stats
        self.experiment_time = experiment_time or self.experiment_time

    def _generate_composite(self, **kwargs):
        number_of_processes = kwargs.get('number_of_processes', self.number_of_processes)
        number_of_variables = kwargs.get('number_of_variables', self.number_of_variables)
        number_of_stores = kwargs.get('number_of_stores', self.number_of_stores)
        process_sleep = kwargs.get('process_sleep', self.process_sleep)

        composer = ManyVariablesComposite({
            'number_of_processes': number_of_processes,
            'number_of_variables': number_of_variables,
            'number_of_stores': number_of_stores,
            'process_sleep': process_sleep,
        })

        self.composite = composer.generate(**kwargs)

    def _initialize_experiment(self, **kwargs):
        self.experiment = Engine(
            processes=self.composite['processes'],
            topology=self.composite['topology'],
            **kwargs)

    def _run_experiment(self, **kwargs):
        self.experiment.update(kwargs['experiment_time'])

    def _get_emitter_data(self, **kwargs):
        data = self.experiment.emitter.get_data()
        return data

    def _get_emitter_timeseries(self, **kwargs):
        timeseries = self.experiment.emitter.get_timeseries()
        return timeseries

    def _profile_method(self, method, **kwargs):
        print_top_stats = kwargs.get('print_top_stats', self.print_top_stats)
        profiler = cProfile.Profile()
        profiler.enable()
        method(**kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler)
        if print_top_stats:
            stats.sort_stats(SortKey.TIME).print_stats(print_top_stats)
        return stats

    def run_profile(self):

        print('GENERATE COMPOSITE')
        self._profile_method(self._generate_composite)

        print('INITIALIZE EXPERIMENT')
        self._profile_method(self._initialize_experiment)

        print('RUN EXPERIMENT')
        self._profile_method(self._run_experiment, experiment_time=self.experiment_time)

        print('GET EMITTER DATA')
        self._profile_method(self._get_emitter_data)

    def profile_communication_latency(self, print_report=True):

        self._generate_composite()
        self._initialize_experiment(display_info=False)

        # profile the experiment
        stats = self._profile_method(
            self._run_experiment,
            experiment_time=self.experiment_time,
            print_top_stats=None)

        # get next_update runtime
        next_update_amount = ("next_update",)
        width, list = stats.get_print_list(next_update_amount)
        cc, nc, tt, ct, callers = stats.stats[list[0]]
        process_update_time = ct

        # get total runtime
        experiment_time = stats.total_tt

        # # print stats
        # stats.print_stats("large_experiment")
        # stats.print_stats("next_update")

        # analyze
        store_update_time = experiment_time - process_update_time

        if print_report:
            print(f"TOTAL EXPERIMENT TIME: {experiment_time}")
            print(f"PROCESS NEXT_UPDATE: {process_update_time}")
            print(f"STORE UPDATE: {store_update_time}")

        return process_update_time, store_update_time

    def run_scan(
            self,
            processes_range=None,
            variables_range=None,
            stores_range=None,
    ):
        processes_range = processes_range or self.processes_range
        variables_range = variables_range or self.variables_range
        stores_range = stores_range or self.stores_range

        saved_stats = {}
        for n_processes in processes_range:
            for n_vars in variables_range:
                for n_stores in stores_range:
                    self.set_parameters(
                        number_of_processes=n_processes,
                        number_of_variables=n_vars,
                        number_of_stores=n_stores)

                    # run experiment
                    process_update_time, store_update_time = \
                        self.profile_communication_latency(print_report=False)

                    # save data
                    saved_stats[(n_processes, n_vars, n_stores)] = (
                        process_update_time, store_update_time)

        # import ipdb;
        # ipdb.set_trace()
        return saved_stats



if __name__ == '__main__':
    # run_large_initial_emit()
    # test_runtime_profile()

    sim = ComplexModelSim()
    sim.from_cli()
