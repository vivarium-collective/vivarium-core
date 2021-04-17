"""
========
Emitters
========

Emitters log configuration data and time-series data somewhere.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from pymongo import MongoClient

from vivarium.library.units import remove_units
from vivarium.library.dict_utils import (
    value_in_embedded_dict,
    make_path_dict,
)
from vivarium.core.process import deserialize_value

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


def get_emitter(config: Optional[Dict[str, str]]) -> 'Emitter':
    """Construct an Emitter using the provided config.

    The available Emitter type names and their classes are:

    * ``database``: :py:class:`DatabaseEmitter`
    * ``null``: :py:class:`NullEmitter`
    * ``print``: :py:class:`Emitter`, prints to stdout
    * ``timeseries``: :py:class:`TimeSeriesEmitter`

    Arguments:
        config (dict): Required keys:
            * 'type': The emitter type name (e.g. 'database').

    Returns:
        Emitter: A new Emitter instance.
    """

    if config is None:
        config = {}
    emitter_type = config.get('type', 'print')

    if emitter_type == 'database':
        emitter: Emitter = DatabaseEmitter(config)
    elif emitter_type == 'null':
        emitter = NullEmitter(config)
    elif emitter_type == 'timeseries':
        emitter = TimeSeriesEmitter(config)
    else:
        emitter = Emitter(config)

    return emitter


def path_timeseries_from_data(data: dict) -> dict:
    """Does something with timeseries data."""
    embedded_timeseries = timeseries_from_data(data)
    return path_timeseries_from_embedded_timeseries(embedded_timeseries)


def path_timeseries_from_embedded_timeseries(embedded_timeseries: dict) -> dict:
    """Does something with timeseries data."""
    times_vector = embedded_timeseries['time']
    path_timeseries = make_path_dict(
        {key: val for key, val in embedded_timeseries.items() if key != 'time'})
    path_timeseries['time'] = times_vector
    return path_timeseries


def time_indexed_timeseries_from_data(data: Dict[float, Any]) -> dict:
    """Does something with timeseries data."""
    times_vector = list(data.keys())
    embedded_timeseries: dict = {}
    for time_index, value in enumerate(data.values()):
        if isinstance(value, dict):
            embedded_timeseries = value_in_embedded_dict(
                value, embedded_timeseries, time_index)
    embedded_timeseries['time'] = times_vector
    return embedded_timeseries


def timeseries_from_data(data: dict) -> dict:
    """Does something with timeseries data."""
    times_vector = list(data.keys())
    embedded_timeseries: dict = {}
    for value in data.values():
        if isinstance(value, dict):
            embedded_timeseries = value_in_embedded_dict(
                value, embedded_timeseries)

    embedded_timeseries['time'] = times_vector
    return embedded_timeseries


class Emitter:
    """
    Emit data to stdout
    """
    def __init__(self, config: Dict[str, str]) -> None:
        self.config = config

    def emit(self, data: Dict[str, Any]) -> None:
        print(data)

    def get_data(self) -> dict:
        return {}

    def get_data_deserialized(self) -> Any:
        return deserialize_value(self.get_data())

    def get_data_unitless(self) -> Any:
        return remove_units(self.get_data_deserialized())

    def get_path_timeseries(self) -> dict:
        return path_timeseries_from_data(self.get_data_deserialized())

    def get_timeseries(self) -> dict:
        return timeseries_from_data(self.get_data_deserialized())


class NullEmitter(Emitter):
    """
    Don't emit anything
    """
    def emit(self, data: Dict[str, Any]) -> None:
        pass


class TimeSeriesEmitter(Emitter):
    """
    Accumulate the timeseries history portion of the "emitted" data to a table
    in RAM.
    """

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.saved_data: Dict[float, Dict[str, Any]] = {}

    def emit(self, data: Dict[str, Any]) -> None:
        """
        Emit the timeseries history portion of `data`, which is
        `data['data'] if data['table'] == 'history'` and put it at
        `data['data']['time']` in the history.
        """
        if data['table'] == 'history':
            emit_data = data['data']
            time = emit_data.pop('time')  # TODO: OK to modify caller's dict?
            self.saved_data[time] = emit_data

    def get_data(self) -> dict:
        """ Return the accumulated timeseries history of "emitted" data. """
        return self.saved_data


class DatabaseEmitter(Emitter):
    """
    Emit data to a mongoDB database

    Example:

    >>> config = {
    ...     'host': 'localhost:27017',
    ...     'database': 'DB_NAME',
    ... }
    >>> # The line below works only if you have to have 27017 open locally
    >>> # emitter = DatabaseEmitter(config)
    """
    client = None
    default_host = 'localhost:27017'

    @classmethod
    def create_indexes(cls, table: Any, columns: List[str]) -> None:
        """Create the listed column indexes for the given DB table."""
        for column in columns:
            table.create_index(column)

    def __init__(self, config: Dict[str, str]) -> None:
        """config may have 'host' and 'database' items."""
        # TODO(jerry): Will this create the DB tables or does it fail if they
        #  don't already exist?
        super().__init__(config)
        self.experiment_id = config.get('experiment_id')

        # create singleton instance of mongo client
        if DatabaseEmitter.client is None:
            DatabaseEmitter.client = MongoClient(
                config.get('host', self.default_host))

        self.db = getattr(self.client, config.get('database', 'simulations'))
        self.history = getattr(self.db, 'history')
        self.configuration = getattr(self.db, 'configuration')
        self.phylogeny = getattr(self.db, 'phylogeny')
        self.create_indexes(self.history, HISTORY_INDEXES)
        self.create_indexes(self.configuration, CONFIGURATION_INDEXES)
        self.create_indexes(self.phylogeny, CONFIGURATION_INDEXES)

    def emit(self, data: Dict[str, Any]) -> None:
        emit_data: dict = data['data']
        emit_data['experiment_id'] = self.experiment_id
        # TODO(jerry): Should this pop('table') from emit_data?
        table = getattr(self.db, data['table'])
        table.insert_one(emit_data)

    def get_data(self) -> dict:
        return get_history_data_db(self.history, self.experiment_id)


def get_history_data_db(
        history_collection: Any, experiment_id: Any) -> Dict[float, dict]:
    """Query MongoDB for history data."""
    query = {'experiment_id': experiment_id}
    raw_data = list(history_collection.find(query))
    data = {}
    for datum in raw_data:
        time = datum['time']
        data[time] = {
            key: value for key, value in datum.items()
            if key not in ['_id', 'experiment_id', 'time']}
    return data


def get_atlas_client(secrets_path: str) -> Any:
    """Open a MongoDB client using the named secrets config JSON file."""
    with open(secrets_path, 'r') as f:
        secrets = json.load(f)
    emitter_config = get_atlas_database_emitter_config(
        **secrets['database'])
    uri = emitter_config['host']
    client = MongoClient(uri)
    return client[emitter_config['database']]


def get_local_client(host: str, port: Any, database_name: str) -> Any:
    """Open a MongoDB client onto the given host, port, and DB."""
    client = MongoClient('{}:{}'.format(host, port))
    return client[database_name]


def data_from_database(experiment_id: str, client: Any) -> Tuple[dict, Any]:
    """Fetch something from a MongoDB."""
    # Retrieve environment config
    config_collection = client.configuration
    experiment_config = config_collection.find_one({
        'experiment_id': experiment_id,
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
    return data, experiment_config


def data_to_database(
        data: Dict[float, dict], environment_config: Any, client: Any) -> Any:
    """Insert something into a MongoDB."""
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
    username: str, password: str, cluster_subdomain: Any, database: str
) -> Dict[str, Any]:
    """Construct an Emitter config for a MongoDB on the Atlas service."""
    username = quote_plus(username)
    password = quote_plus(password)
    database = quote_plus(database)

    uri = (
        "mongodb+srv://{}:{}@{}.mongodb.net/"
        "?retryWrites=true&w=majority"
    ).format(username, password, cluster_subdomain)
    return {
        'type': 'database',
        'host': uri,
        'database': database,
    }


def emit_environment_config(
        environment_config: Dict[str, Any], emitter: Emitter) -> None:
    """Emit a multibody bounds environment config to the given Emitter."""
    config = {
        'bounds': environment_config['multibody']['bounds'],
        'type': 'environment_config',
    }
    emitter.emit({
        'data': config,
        'table': 'configuration',
    })
