"""
Experiment to test maximum BSON document size with MongoDB emitter
"""
import uuid
import random
import cProfile, pstats
from pstats import SortKey

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
        'variables': [],
    }
    def __init__(self, parameters=None):
        super().__init__(parameters)
        # make a bunch of variables
        random_variables = [key for key in range(self.parameters['number_of_variables'])]
        super().__init__({'variables': random_variables})
    def ports_schema(self):
        return {
            'port': {
                variable: {
                    '_default': random.random(),
                    '_emit': True
                } for variable in self.parameters['variables']}}
    def next_update(self, timestep, states):
        return {
            'port': {
                variable: random.random()
                for variable in self.parameters['variables']}}

class ManyVariablesComposite(Composer):
    defaults = {
        'number_of_processes': 10,
        'number_of_variables': 10}
    def __init__(self, config=None):
        super().__init__(config)
        self.process_ids = [
            f'process_{key}'
            for key in range(self.config['number_of_processes'])]
    def generate_processes(self, config):
        # make a bunch of processes
        return {
            process_id: ManyVariablesProcess({
                'name': process_id,
                'number_of_variables': self.config['number_of_variables']})
            for process_id in self.process_ids}
    def generate_topology(self, config):
        return {
            process_id: {'port': ('store', process_id,)}
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


def test_model_complexity():
    config = {
        'number_of_processes': 10,
        'number_of_variables': 10}

    composer = ManyVariablesComposite(config)
    composite = composer.generate()

    experiment = Engine(
        processes=composite['processes'],
        topology=composite['topology'],
    )

    # run the experiment
    experiment.update(4)

    # get the data
    timeseries = experiment.emitter.get_timeseries()


class ComplexModelSim:

    def __init__(self):
        self.number_of_processes = 10
        self.number_of_variables = 10

        # initialize
        self.initialize_composite()
        self.generate_composite()
        self.initialize_experiment()

    def initialize_composite(self):
        self.composer = ManyVariablesComposite({
            'number_of_processes': self.number_of_processes,
            'number_of_variables': self.number_of_variables})

    def generate_composite(self):
        self.composite = self.composer.generate()

    def initialize_experiment(self):
        self.experiment = Engine(
            processes=self.composite['processes'],
            topology=self.composite['topology'])

    def run_experiment(self, total_time=10):
        self.experiment.update(total_time)

    def get_emitter_data(self):
        self.data = self.experiment.emitter.get_timeseries()

    def profile_method(self, method):
        profiler = cProfile.Profile()
        profiler.enable()
        method()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats(SortKey.TIME).print_stats(10)
        stats.print_stats(10)


    def run_profile(self):
        print('INITIALIZE COMPOSITE')
        self.profile_method(self.initialize_composite)

        print('GENERATE COMPOSITE')
        self.profile_method(self.generate_composite)

        print('INITIALIZE EXPERIMENT')
        self.profile_method(self.initialize_experiment)

        print('RUN EXPERIMENT')
        self.profile_method(self.run_experiment)

        print('GET EMITTER DATA')
        self.profile_method(self.get_emitter_data)





if __name__ == '__main__':
    # run_large_initial_emit()
    # test_runtime_profile()
    # profile_model_complexity()
    ComplexModelSim().run_profile()