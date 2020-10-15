from __future__ import absolute_import, division, print_function

from pymongo import MongoClient
from confluent_kafka import Producer
import json
from urllib.parse import quote_plus

from vivarium.library.dict_utils import (
    value_in_embedded_dict, make_path_dict)

HISTORY_INDEXES = [
    'time',
    'type',
    'simulation_id',
    'experiment_id']

CONFIGURATION_INDEXES = [
    'type',
    'simulation_id',
    'experiment_id']

SECRETS_PATH = 'secrets.json'


def delivery_report(err, msg):
    """
    This is a utility method passed to the Kafka Producer to handle the delivery
    of messages sent using `send(topic, message)`.
    """
    if err is not None:
        print('message delivery failed: {}'.format(msg))
        print('failed message: {}'.format(err))

def create_indexes(table, columns):
    '''Create all of the necessary indexes for the given table name.'''
    for column in columns:
        table.create_index(column)

def get_emitter(config):
    '''Get an emitter based on the provided config.

    The available emitter type names and their classes are:

    * ``kafka``: :py:class:`KafkaEmitter`
    * ``database``: :py:class:`DatabaseEmitter`
    * ``null``: :py:class:`NullEmitter`
    * ``timeseries``: :py:class:`TimeSeriesEmitter`

    Arguments:
        config (dict): Requires three keys:
            * type: Type of emitter ('kafka' for a kafka emitter).
            * emitter: Any configuration the emitter type requires to initialize.
            * keys: A list of state keys to emit for each state label.

    Returns:
        Emitter: An instantiated emitter.
    '''

    if config is None:
        config = {'type': 'print'}
    emitter_type = config.get('type', 'print')

    if emitter_type == 'kafka':
        emitter = KafkaEmitter(config)
    elif emitter_type == 'database':
        emitter = DatabaseEmitter(config)
    elif emitter_type == 'null':
        emitter = NullEmitter(config)
    elif emitter_type == 'timeseries':
        emitter = TimeSeriesEmitter(config)
    else:
        emitter = Emitter(config)

    return emitter

def configure_emitter(config, processes, topology):
    emitter_config = config.get('emitter', {})
    emitter_config['experiment_id'] = config.get('experiment_id')
    emitter_config['simulation_id'] = config.get('simulation_id')
    return get_emitter(emitter_config)

def path_timeseries_from_data(data):
    embedded_timeseries = timeseries_from_data(data)
    return path_timeseries_from_embedded_timeseries(embedded_timeseries)

def path_timeseries_from_embedded_timeseries(embedded_timeseries):
    times_vector = embedded_timeseries['time']
    path_timeseries = make_path_dict({key: value for key, value in embedded_timeseries.items() if key != 'time'})
    path_timeseries['time'] = times_vector
    return path_timeseries

def time_indexed_timeseries_from_data(data):
    times_vector = list(data.keys())
    embedded_timeseries = {}
    for time_index, (time, value) in enumerate(data.items()):
        if isinstance(value, dict):
            embedded_timeseries = value_in_embedded_dict(value, embedded_timeseries, time_index)
        else:
            pass
    embedded_timeseries['time'] = times_vector
    return embedded_timeseries

def timeseries_from_data(data):
    times_vector = list(data.keys())
    embedded_timeseries = {}
    for time, value in data.items():
        if isinstance(value, dict):
            embedded_timeseries = value_in_embedded_dict(value, embedded_timeseries)
        else:
            pass

    embedded_timeseries['time'] = times_vector
    return embedded_timeseries


class Emitter(object):
    '''
    Emit data to terminal
    '''
    def __init__(self, config):
        self.config = config

    def emit(self, data):
        print(data)

    def get_data(self):
        return {}

    def get_path_timeseries(self):
        return path_timeseries_from_data(self.get_data())

    def get_timeseries(self):
        return timeseries_from_data(self.get_data())


class NullEmitter(Emitter):
    '''
    Don't emit anything
    '''
    def emit(self, data):
        pass


class TimeSeriesEmitter(Emitter):

    def __init__(self, config):
        super().__init__(config)
        self.saved_data = {}

    def emit(self, data):
        # save history data
        if data['table'] == 'history':
            emit_data = data['data']
            time = emit_data.pop('time')
            self.saved_data[time] = emit_data

    def get_data(self):
        return self.saved_data


class KafkaEmitter(Emitter):
    '''
    Emit data to kafka

    Example:

    >>> config = {
    ...     'host': 'localhost:9092',
    ...     'topic': 'EMIT',
    ... }
    >>> emitter = KafkaEmitter(config)
    '''
    def __init__(self, config):
        super().__init__(config)
        self.producer = Producer({
            'bootstrap.servers': self.config['host']})

    def emit(self, data):
        encoded = json.dumps(data, ensure_ascii=False).encode('utf-8')

        self.producer.produce(
            self.config['topic'],
            encoded,
            callback=delivery_report)

        self.producer.flush(timeout=0.1)


class DatabaseEmitter(Emitter):
    '''
    Emit data to a mongoDB database

    Example:

    >>> config = {
    ...     'host': 'localhost:27017',
    ...     'database': 'DB_NAME',
    ... }
    >>> # The line below works only if you have to have 27017 open locally
    >>> # emitter = DatabaseEmitter(config)
    '''
    client = None
    default_host = 'localhost:27017'

    def __init__(self, config):
        super().__init__(config)
        self.experiment_id = config.get('experiment_id')

        # create singleton instance of mongo client
        if DatabaseEmitter.client is None:
            DatabaseEmitter.client = MongoClient(config.get('host', self.default_host))

        self.db = getattr(self.client, config.get('database', 'simulations'))
        self.history = getattr(self.db, 'history')
        self.configuration = getattr(self.db, 'configuration')
        self.phylogeny = getattr(self.db, 'phylogeny')
        create_indexes(self.history, HISTORY_INDEXES)
        create_indexes(self.configuration, CONFIGURATION_INDEXES)
        create_indexes(self.phylogeny, CONFIGURATION_INDEXES)

    def emit(self, data_config):
        data = data_config['data']
        data.update({
            'experiment_id': self.experiment_id})
        table = getattr(self.db, data_config['table'])
        table.insert_one(data)

    def get_data(self):
        return get_history_data_db(self.history, self.experiment_id)


def get_history_data_db(history_collection, experiment_id):
    query = {'experiment_id': experiment_id}
    raw_data = history_collection.find(query)
    raw_data = list(raw_data)
    data = {}
    for datum in raw_data:
        time = datum['time']
        data[time] = {
            key: value for key, value in datum.items()
            if key not in ['_id', 'experiment_id', 'time']}
    return data


def get_atlas_client(secrets_path):
    with open(secrets_path, 'r') as f:
        secrets = json.load(f)
    emitter_config = get_atlas_database_emitter_config(
        **secrets['database'])
    uri = emitter_config['host']
    client = MongoClient(uri)
    return client[emitter_config['database']]


def get_local_client(host, port, database_name):
    client = MongoClient('{}:{}'.format(host, port))
    return client[database_name]


def data_from_database(experiment_id, client):
    # Retrieve environment config
    config_collection = client.configuration
    environment_config = config_collection.find_one({
        'experiment_id': experiment_id,
        'type': 'environment_config',
    })

    # Retrieve timepoint data
    history_collection = client.history

    unique_time_objs = history_collection.aggregate([
        {
            '$match': {
                'experiment_id': experiment_id
            }
        }, {
            '$group': {
                '_id': {
                    'time': '$time'
                },
                'id': {
                    '$first': '$_id'
                }
            }
        }, {
            '$sort': {
                '_id.time': 1
            }
        },
    ])
    unique_time_ids = [
        obj['id'] for obj in unique_time_objs
    ]
    data_cursor = history_collection.find({
        '_id': {
            '$in': unique_time_ids
        }
    }).sort('time')
    raw_data = list(data_cursor)

    # Reshape data
    data = {
        timepoint_dict['time']: {
            key: val
            for key, val in timepoint_dict.items()
            if key != 'time'
        }
        for timepoint_dict in raw_data
    }
    return data, environment_config


def data_to_database(data, environment_config, client):
    history_collection = client.history
    reshaped_data = []
    for time, timepoint_data in data.items():
        # Since time is the dictionary key, it has to be a string for
        # JSON/BSON compatibility. But now that we're uploading it, we
        # want it to be a float for fast searching.
        reshaped_entry = {'time': float(time)}
        for key, val in timepoint_data.items():
            if key not in ('_id', 'time'):
                reshaped_entry[key] = val
        reshaped_data.append(reshaped_entry)
    history_collection.insert_many(reshaped_data)

    config_collection = client.configuration
    config_collection.insert_one(environment_config)


def get_atlas_database_emitter_config(
    username, password, cluster_subdomain, database
):
    username = quote_plus(username)
    password = quote_plus(password)
    database = quote_plus(database)

    uri = (
        "mongodb+srv://{}:{}@{}.mongodb.net/"
        + "?retryWrites=true&w=majority"
    ).format(username, password, cluster_subdomain)
    return {
        'type': 'database',
        'host': uri,
        'database': database,
    }


def emit_environment_config(environment_config, emitter):
    config = {
        'bounds': environment_config['multibody']['bounds'],
        'type': 'environment_config',
    }
    emitter.emit({
        'data': config,
        'table': 'configuration',
    })
