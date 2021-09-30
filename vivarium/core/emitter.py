"""
========
Emitters
========

Emitters log configuration data and time-series data somewhere.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from pymongo import MongoClient

from vivarium.library.units import remove_units
from vivarium.library.dict_utils import (
    value_in_embedded_dict,
    make_path_dict,
    deep_merge,
)
from vivarium.library.topology import assoc_path
from vivarium.core.serialize import deserialize_value
from vivarium.core.registry import emitter_registry

MONGO_DOCUMENT_LIMIT = 1e7

HISTORY_INDEXES = [
    'time',
    'type',
    'simulation_id',
    'experiment_id',
]

CONFIGURATION_INDEXES = [
    'type',
    'simulation_id',
    'experiment_id',
]

SECRETS_PATH = 'secrets.json'


def size_of(emit_data: Any) -> int:
    return len(str(emit_data))


def breakdown_data(
        limit: float,
        data: Any,
        path: Tuple = (),
        size: float = None,
) -> list:
    size = size or size_of(data)
    if size > limit:
        if isinstance(data, dict):
            output = []
            subsizes = {}
            total = 0
            for key, subdata in data.items():
                subsizes[key] = size_of(subdata)
                total += subsizes[key]

            order = sorted(
                subsizes.items(),
                key=lambda item: item[1],
                reverse=True)

            remaining = total
            index = 0
            large_keys = []
            while remaining > limit and index < len(order):
                key, subsize = order[index]
                large_keys.append(key)
                remaining -= subsize
                index += 1

            for large_key in large_keys:
                subdata = breakdown_data(
                    limit,
                    data[large_key],
                    path=path + (large_key,),
                    size=subsizes[large_key])

                try:
                    output.extend(subdata)
                except ValueError:
                    print(f'data can not be broken down to size '
                          f'{limit}: {data[large_key]}')

            pruned = {
                key: value
                for key, value in data.items()
                if key not in large_keys}
            output.append((path, pruned))
            return output

        print('value is too large to emit, ignoring data')
        return []

    return [(path, data)]


def get_emitter(config: Optional[Dict[str, str]]) -> 'Emitter':
    """Construct an Emitter using the provided config.

    The available Emitter type names and their classes are:

    * ``database``: :py:class:`DatabaseEmitter`
    * ``null``: :py:class:`NullEmitter`
    * ``print``: :py:class:`Emitter`, prints to stdout
    * ``timeseries``: :py:class:`RAMEmitter`

    Arguments:
        config: Must comtain the ``type`` key, which specifies the emitter
            type name (e.g. ``database``).

    Returns:
        A new Emitter instance.
    """

    if config is None:
        config = {}
    emitter_type = config.get('type', 'print')
    emitter: Emitter = emitter_registry.access(emitter_type)(config)
    return emitter


def path_timeseries_from_data(data: dict) -> dict:
    """Convert from :term:`raw data` to a :term:`path timeseries`."""
    embedded_timeseries = timeseries_from_data(data)
    return path_timeseries_from_embedded_timeseries(embedded_timeseries)


def path_timeseries_from_embedded_timeseries(embedded_timeseries: dict) -> dict:
    """Convert an :term:`embedded timeseries` to a :term:`path timeseries`."""
    times_vector = embedded_timeseries['time']
    path_timeseries = make_path_dict(
        {key: val for key, val in embedded_timeseries.items() if key != 'time'})
    path_timeseries['time'] = times_vector
    return path_timeseries


def timeseries_from_data(data: dict) -> dict:
    """Convert :term:`raw data` to an :term:`embedded timeseries`."""
    times_vector = list(data.keys())
    embedded_timeseries: dict = {}
    for value in data.values():
        if isinstance(value, dict):
            embedded_timeseries = value_in_embedded_dict(
                value, embedded_timeseries)

    embedded_timeseries['time'] = times_vector
    return embedded_timeseries


class Emitter:
    def __init__(self, config: Dict[str, str]) -> None:
        """Base class for emitters.

        This emitter simply emits to STDOUT.

        Args:
            config: Emitter configuration.
        """
        self.config = config

    def emit(self, data: Dict[str, Any]) -> None:
        """Emit data.

        Args:
            data: The data to emit. This gets called by the Vivarium
                engine with a snapshot of the simulation state.
        """
        print(data)

    def get_data(self) -> dict:
        """Get the emitted data.

        Returns:
            The data that has been emitted to the database in the
            :term:`raw data` format. For this particular class, an empty
            dictionary is returned.
        """
        return {}

    def get_data_deserialized(self) -> Any:
        """Get the emitted data with variable values deserialized.

        Returns:
            The data that has been emitted to the database in the
            :term:`raw data` format. Before being returned, serialized
            values in the data are deserialized.
        """
        return deserialize_value(self.get_data())

    def get_data_unitless(self) -> Any:
        """Get the emitted data with units stripped from variable values.

        Returns:
            The data that has been emitted to the database in the
            :term:`raw data` format. Before being returned, units are
            stripped from values.
        """
        return remove_units(self.get_data_deserialized())

    def get_path_timeseries(self) -> dict:
        """Get the deserialized data as a :term:`path timeseries`.

        Returns:
            The deserialized emitted data, formatted as a
            :term:`path timeseries`.
        """
        return path_timeseries_from_data(self.get_data_deserialized())

    def get_timeseries(self) -> dict:
        """Get the deserialized data as an :term:`embedded timeseries`.

        Returns:
            The deserialized emitted data, formatted as an
            :term:`embedded timeseries`.
        """
        return timeseries_from_data(self.get_data_deserialized())


class NullEmitter(Emitter):
    """
    Don't emit anything
    """
    def emit(self, data: Dict[str, Any]) -> None:
        pass


class RAMEmitter(Emitter):
    """
    Accumulate the timeseries history portion of the "emitted" data to a table
    in RAM.
    """

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.saved_data: Dict[float, Dict[str, Any]] = {}

    def emit(self, data: Dict[str, Any]) -> None:
        """
        Emit the timeseries history portion of ``data``, which is
        ``data['data'] if data['table'] == 'history'`` and put it at
        ``data['data']['time']`` in the history.
        """
        if data['table'] == 'history':
            emit_data = data['data']
            time = emit_data['time']
            self.saved_data[time] = {
                key: value for key, value in emit_data.items()
                if key not in ['time']}

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

    def __init__(self, config: Dict[str, Any]) -> None:
        """config may have 'host' and 'database' items."""
        super().__init__(config)
        self.experiment_id = config.get('experiment_id')
        self.emit_limit = config.get('emit_limit', MONGO_DOCUMENT_LIMIT)

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
        table_id = data['table']
        table = getattr(self.db, table_id)
        emit_data = {
            key: value for key, value in data.items()
            if key not in ['table']}
        emit_data['experiment_id'] = self.experiment_id
        self.write_emit(table, emit_data)

    def write_emit(self, table: Any, emit_data: Dict[str, Any]) -> None:
        """Check that data size is less than emit limit.

        Break up large emits into smaller pieces and emit them individually
        """
        data = emit_data.pop('data')
        broken_down_data = breakdown_data(self.emit_limit, data)
        assembly_id = str(uuid.uuid4())
        for (path, datum) in broken_down_data:
            d = dict(emit_data)
            assoc_path(d, ('data',) + path, datum)
            d['assembly_id'] = assembly_id
            table.insert_one(d)

    def get_data(self) -> dict:
        return get_history_data_db(self.history, self.experiment_id)


def get_experiment_database(
        port: Any = 27017,
        database_name: str = 'simulations'
) -> Any:
    """Get a database object.

    Args:
        port: Port number of database. This can usually be left as the
            default.
        database_name: Name of the database table. This can usually be
            left as the default.

    Returns:
        The database object.
    """
    config = {
        'host': '{}:{}'.format('localhost', port),
        'database': database_name}
    emitter = DatabaseEmitter(config)
    db = emitter.db
    return db


def delete_experiment_from_database(
        experiment_id: str,
        port: Any = 27017,
        database_name: str = 'simulations'
) -> None:
    """Delete an experiment's data from a database.

    Args:
        experiment_id: Identifier of experiment.
        port: Port number of database. This can usually be left as the
            default.
        database_name: Name of the database table. This can usually be
            left as the default.
    """
    db = get_experiment_database(port, database_name)
    query = {'experiment_id': experiment_id}
    db.history.delete_many(query)
    db.configuration.delete_many(query)


def assemble_data(data: list) -> dict:
    """re-assemble data"""
    assembly: dict = {}
    for datum in data:
        if 'assembly_id' in datum:
            assembly_id = datum['assembly_id']
            if assembly_id not in assembly:
                assembly[assembly_id] = {}
            deep_merge(assembly[assembly_id], datum['data'])
        else:
            assembly_id = str(uuid.uuid4())
            assembly[assembly_id] = datum['data']
    return assembly


def get_history_data_db(
        history_collection: Any, experiment_id: Any) -> Dict[float, dict]:
    """Query MongoDB for history data."""
    query = {'experiment_id': experiment_id}
    raw_data = list(history_collection.find(query))

    # re-assemble data
    assembly = assemble_data(raw_data)

    # restructure by time
    data = {}
    for datum in assembly.values():
        time = datum['time']
        data[time] = {
            key: value for key, value in datum.items()
            if key not in ['_id', 'time']}

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
    query = {'experiment_id': experiment_id}
    experiment_configs = list(config_collection.find(query))

    # Re-assemble experiment_config
    experiment_assembly = assemble_data(experiment_configs)
    assert len(experiment_assembly) == 1
    assembly_id = list(experiment_assembly.keys())[0]
    experiment_config = experiment_assembly[assembly_id]

    # Retrieve timepoint data
    history = client.history
    data = get_history_data_db(history, experiment_id)

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


def test_breakdown() -> None:
    data = {
        'a': [1, 2, 3],
        'b': {
            'X': [1, 2, 3, 4, 5],
            'Y': [1, 2, 3, 4, 5, 6],
            'Z': [5]}}

    output = breakdown_data(20, data)
    assert output == [
        (('b', 'Y'), [1, 2, 3, 4, 5, 6]),
        (('b',), {'X': [1, 2, 3, 4, 5], 'Z': [5]}),
        ((), {'a': [1, 2, 3]})]


if __name__ == '__main__':
    test_breakdown()
