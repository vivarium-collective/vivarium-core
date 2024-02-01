"""
Experiment to test maximum BSON document size with MongoDB emitter
"""
import uuid
import random

from vivarium.core.engine import Engine
from vivarium.core.process import Process
from vivarium.core.composer import Composer
from vivarium.core.emitter import (
    get_experiment_database,
    data_from_database,
    delete_experiment_from_database)


class ManyParametersProcess(Process):
    defaults = {
        'number_of_parameters': 100,
        'number_of_ports': 1}
    def __init__(self, parameters=None):
        # make a bunch of parameters
        random_parameters = {
            key: random.random()
            for key in range(parameters['number_of_parameters'])}
        super().__init__({**parameters, **random_parameters})
    def ports_schema(self):
        return {'port': {
            str(i): {'_default': 0, '_emit': True}
            for i in range(self.parameters['number_of_ports'])}}
    def next_update(self, timestep, states):
        return {'port': {str(i): 1
            for i in range(self.parameters['number_of_ports'])}}


class ManyParametersComposite(Composer):
    defaults = {
        'number_of_processes': 10,
        'number_of_parameters': 100,
        'number_of_ports': 1}
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
                'number_of_parameters': self.config['number_of_parameters'],
                'number_of_ports': self.config['number_of_ports']})
            for process_id in self.process_ids}
    def generate_topology(self, config):
        return {
            process_id: {'port': ('store', process_id,)}
            for process_id in self.process_ids}


def run_large_experiment(config):
    """
    This experiment runs a large experiment to test the database emitter.
    This requires MongoDB to be configured and running.
    """
    composer = ManyParametersComposite(config)
    composite = composer.generate()

    settings = {
        'experiment_name': 'large database experiment',
        'experiment_id': f'large_{str(uuid.uuid4())}',
        'emitter': 'database',
        'emit_config': True
    }

    experiment = Engine(**{
        'processes': composite['processes'],
        'topology': composite['topology'],
        **settings})

    # run the experiment
    experiment.update(1)

    # retrieve the data with data_from_database
    experiment_id = experiment.experiment_id

    # retrieve the data from emitter
    data = experiment.emitter.get_data()
    assert list(data.keys()) == [0.0, 1.0]

    # retrieve the data directly from database
    db = get_experiment_database()
    data, experiment_config = data_from_database(experiment_id, db)
    assert 'processes' in experiment_config
    assert 0.0 in data

    # check keys of emitted data
    state = experiment_config['state']
    store_names = set(str(i) for i in range(config['number_of_ports']))
    store_values = set([1] * config['number_of_ports'])
    assert state.keys() == set(composite.processes) | set(('store',))
    for proc, stores in state['store'].items():
        assert stores.keys() == store_names
        assert set(data[1.0]['store'][proc].values()) == store_values

    # delete the experiment
    delete_experiment_from_database(experiment_id)


def test_large_initial_emit():
    run_large_experiment({
        'number_of_processes': 1000,
        'number_of_parameters': 1000,
        'number_of_ports': 1})


def test_large_sim_emit():
    run_large_experiment({
        'number_of_processes': 1000,
        'number_of_parameters': 1,
        'number_of_ports': 1000})


if __name__ == '__main__':
    test_large_initial_emit()
    test_large_sim_emit()
