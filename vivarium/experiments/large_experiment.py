"""
Experiment to test maximum BSON document size with MongoDB emitter
"""
import uuid
import random

import numpy as np

from vivarium.core.engine import Engine
from vivarium.core.process import Process
from vivarium.core.composer import Composer
from vivarium.core.emitter import (
    get_experiment_database,
    data_from_database,
    data_to_database,
    delete_experiment_from_database)


class ManyParametersProcess(Process):
    defaults = {
        'number_of_parameters': 100,
        'arr_size': 1}
    def __init__(self, parameters=None):
        # make a bunch of parameters
        random_parameters = {
            key: random.random()
            for key in range(parameters['number_of_parameters'])}
        super().__init__({**parameters, **random_parameters})
    def ports_schema(self):
        return {'port': {'_default': np.random.random(
            self.parameters['arr_size']), '_emit': True}}
    def next_update(self, timestep, states):
        return {'port': 1}


class ManyParametersComposite(Composer):
    defaults = {
        'number_of_processes': 10,
        'number_of_parameters': 100,
        'arr_size': 1}
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
                'arr_size': self.config['arr_size']})
            for process_id in self.process_ids}
    def generate_topology(self, config):
        return {
            process_id: {'port': ('store', process_id,)}
            for process_id in self.process_ids}


def test_large_emits():
    """
    This experiment runs a large experiment to test the database emitter.
    This requires MongoDB to be configured and running.
    """
    config = {
        'number_of_processes': 10,
        'number_of_parameters': 100000,
        'arr_size': 150000}

    composer = ManyParametersComposite(config)
    composite = composer.generate()

    settings = {
        'experiment_name': 'large database experiment',
        'experiment_id': f'large_{str(uuid.uuid4())}',
        'emitter': 'database',
        'emit_processes': True
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

    db = get_experiment_database()
    # check that configuration emit was split into sub-documents
    config_cursor = db.configuration.find({'experiment_id': experiment_id})
    config_raw = list(config_cursor)
    assert len(config_raw) > 1
    # check that sim emit was split into sub-documents (2 timesteps)
    history_cursor = db.history.find({'experiment_id': experiment_id})
    history_raw = list(history_cursor)
    assert len(history_raw) > 2

    # retrieve the data directly from database
    data, experiment_config = data_from_database(experiment_id, db)

    # check values of emitted data
    processes = experiment_config['processes']
    assert len(processes) == config['number_of_processes']
    for proc in processes:
        np.testing.assert_allclose(np.array(data[1.0]['store'][proc]),
            np.array(data[0.0]['store'][proc]) + 1)

    # delete the experiment
    delete_experiment_from_database(experiment_id)


def test_query_db():
    """
    This tests the query features of the MongoDB API.
    """
    composer = ManyParametersComposite()
    composite = composer.generate()
    settings = {
        'experiment_name': 'large database experiment',
        'experiment_id': f'large_{str(uuid.uuid4())}',
        'emitter': 'database'
    }
    experiment = Engine(**{
        'processes': composite['processes'],
        'topology': composite['topology'],
        **settings})
    experiment.update(10)

    db = get_experiment_database()

    # test query
    query = [('store', 'process_0'), ('store', 'process_9')]
    data, _ = data_from_database(experiment.experiment_id, db, query)
    assert data[0.0]['store'].keys() == {'process_0', 'process_9'}

    # test func_dict
    func_dict = {
        ('store', 'process_0'): lambda x: 3,
    }
    data, _ = data_from_database(experiment.experiment_id, db, query,
                                 func_dict=func_dict)
    for emit_data in data.values():
        assert emit_data['store']['process_0'] == 3

    # test start and end time
    data, _ = data_from_database(experiment.experiment_id, db,
                                 start_time=1, end_time=5)
    assert data.keys() == {1.0, 2.0, 3.0, 4.0, 5.0}

    # test multiple cpu processes
    data_multi, _ = data_from_database(experiment.experiment_id, db, cpus=2)
    data, _ = data_from_database(experiment.experiment_id, db)
    assert data_multi == data
    delete_experiment_from_database(experiment.experiment_id, cpus=2)


def test_data_to_database():
    data = {
        '0.0': {'data': {'store': 1}, 
                'experiment_id': 'manual_insert'},
        '1.0': {'data': {'store': 2}, 
                'experiment_id': 'manual_insert'},
        '2.0': {'data': {'store': [3.5]}, 
                'experiment_id': 'manual_insert'}
    }
    config = {'experiment_id': 'manual_insert',
              'data': {'store': 1}}
    db = get_experiment_database()
    data_to_database(data, config, db)
    retrieved_data, retrieved_config = data_from_database('manual_insert', db)
    assert retrieved_data == {float(t): val['data'] for t, val in data.items()}
    assert retrieved_config == config['data']
    delete_experiment_from_database('manual_insert')
    db.configuration.delete_many({'experiment_id': 'manual_insert'})


if __name__ == '__main__':
    test_large_emits()
    test_query_db()
    test_data_to_database()
