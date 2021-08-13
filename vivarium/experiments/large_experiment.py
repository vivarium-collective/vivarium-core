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
        start = time.time()
        update = {
                variable: random.random()
                for variable in self.parameters['variables']}
        end = time.time()
        time.sleep(self.parameters['process_sleep'])
        runtime = end - start
        return {
            'port': update,
            'update_timer': runtime,
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
                'update_timer': ('update_timer',),
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

    number_of_processes = 10
    number_of_variables = 10
    process_sleep = 1e-3
    print_top_stats = 4
    experiment_time = 100

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
        args = parser.parse_args()

        if args.profile:
            self.run_profile()
        if args.latency:
            self.profile_communication_latency()

    def set_parameters(
            self,
            number_of_processes=None,
            number_of_variables=None,
            process_sleep=None,
            print_top_stats=None,
            experiment_time=None,
    ):
        self.number_of_processes = number_of_processes or self.number_of_processes
        self.number_of_variables = number_of_variables or self.number_of_variables
        self.process_sleep = process_sleep or self.process_sleep
        self.print_top_stats = print_top_stats or self.print_top_stats
        self.experiment_time = experiment_time or self.experiment_time

    def generate_composite(self, **kwargs):
        self.composer = ManyVariablesComposite({
            'number_of_processes': self.number_of_processes,
            'number_of_variables': self.number_of_variables,
            'process_sleep': self.process_sleep,
        })

        self.composite = self.composer.generate(**kwargs)

    def initialize_experiment(self, **kwargs):
        self.experiment = Engine(
            processes=self.composite['processes'],
            topology=self.composite['topology'],
            **kwargs)

    def run_experiment(self, **kwargs):
        self.experiment.update(kwargs['experiment_time'])

    def get_emitter_data(self, **kwargs):
        self.data = self.experiment.emitter.get_data()
        return self.data

    def get_emitter_timeseries(self, **kwargs):
        self.timeseries = self.experiment.emitter.get_timeseries()
        return self.timeseries

    def profile_method(self, method, **kwargs):
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
        self.profile_method(self.generate_composite)

        print('INITIALIZE EXPERIMENT')
        self.profile_method(self.initialize_experiment)

        print('RUN EXPERIMENT')
        self.profile_method(self.run_experiment, experiment_time=self.experiment_time)

        print('GET EMITTER DATA')
        self.profile_method(self.get_emitter_data)

    def profile_communication_latency(self):

        self.generate_composite()
        self.initialize_experiment()

        print('RUN EXPERIMENT')
        stats = self.profile_method(
            self.run_experiment,
            experiment_time=self.experiment_time,
            print_top_stats=None)

        # self.get_emitter_data()
        self.get_emitter_timeseries()

        # analyze
        experiment_time = stats.total_tt
        process_update_time = self.timeseries['update_timer'][-1]
        store_update_time = experiment_time - process_update_time

        print(f"TOTAL EXPERIMENT TIME: {experiment_time}")
        print(f"PROCESS NEXT_UPDATE: {process_update_time}")
        print(f"STORE UPDATE: {store_update_time}")



if __name__ == '__main__':
    # run_large_initial_emit()
    # test_runtime_profile()

    sim = ComplexModelSim()
    sim.from_cli()
    # sim.run_profile()
    # sim.profile_communication_latency()