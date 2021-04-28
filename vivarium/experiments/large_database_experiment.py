import random
from vivarium.core.experiment import Experiment
from vivarium.core.process import Process, Composer
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
        return {'port': {'variable': {'_default': 0}}}
    def next_update(self, timestep, states):
        return {}


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
            process_id: {'port': ('store',)}
            for process_id in self.process_ids}

def run_large_initial_emit():

    config = {
        'number_of_processes': 1000,
        'number_of_parameters': 1000}

    composer = ManyParametersComposite(config)
    composite = composer.generate()

    settings = {
        'emitter': 'database'
    }

    experiment = Experiment({
        'processes': composite['processes'],
        'topology': composite['topology'],
        **settings})

    # retrieve the data
    experiment_id = experiment.experiment_id
    db = get_experiment_database()
    data, experiment_config = data_from_database(experiment_id, db)

    assert 'processes' in experiment_config
    assert 0.0 in data

    # delete the experiment
    delete_experiment_from_database(experiment_id)


if __name__ == '__main__':
    run_large_initial_emit()
