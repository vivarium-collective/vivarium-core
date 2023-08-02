"""
========
Emitters
========

Emitters log configuration data and time-series data somewhere.
"""

import os
import json
import uuid
import itertools
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from urllib.parse import quote_plus
from concurrent.futures import ProcessPoolExecutor

from pymongo import ASCENDING
from pymongo.errors import DocumentTooLarge
from pymongo.mongo_client import MongoClient
from bson import MinKey, MaxKey

from vivarium.library.units import remove_units
from vivarium.library.dict_utils import (
    value_in_embedded_dict,
    make_path_dict,
    deep_merge_check,
)
from vivarium.library.topology import (
    assoc_path,
    get_in,
    paths_to_dict,
)
from vivarium.core.registry import emitter_registry
from vivarium.core.serialize import (
    make_fallback_serializer_function,
    serialize_value,
    deserialize_value)

HISTORY_INDEXES = [
    'data.time',
    [('experiment_id', ASCENDING),
     ('data.time', ASCENDING),
     ('_id', ASCENDING)],
]

CONFIGURATION_INDEXES = [
    'experiment_id',
]

SECRETS_PATH = 'secrets.json'


def breakdown_data(
        limit: float,
        data: Any,
        path: Tuple = (),
        size: Optional[float] = None,
) -> list:
    size = size or len(str(data))
    if size > limit:
        if isinstance(data, dict):
            output = []
            subsizes = {}
            total = 0
            for key, subdata in data.items():
                subsizes[key] = len(str(subdata))
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

        print(f'Data at {path} is too large, skipped: {size} > {limit}')
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
        config: Must contain the ``type`` key, which specifies the emitter
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
        _ = self # Silence pylint no-self-use
        print(data)

    def get_data(self, query: Optional[list] = None) -> dict:
        """Get the emitted data.

        Returns:
            The data that has been emitted to the database in the
            :term:`raw data` format. For this particular class, an empty
            dictionary is returned.
        """
        _ = query
        _ = self # Silence pylint no-self-use
        return {}

    def get_data_deserialized(self, query: Optional[list] = None) -> Any:
        """Get the emitted data with variable values deserialized.

        Returns:
            The data that has been emitted to the database in the
            :term:`raw data` format. Before being returned, serialized
            values in the data are deserialized.
        """
        return deserialize_value(self.get_data(query))

    def get_data_unitless(self, query: Optional[list] = None) -> Any:
        """Get the emitted data with units stripped from variable values.

        Returns:
            The data that has been emitted to the database in the
            :term:`raw data` format. Before being returned, units are
            stripped from values.
        """
        return remove_units(self.get_data_deserialized(query))

    def get_path_timeseries(self, query: Optional[list] = None) -> dict:
        """Get the deserialized data as a :term:`path timeseries`.

        Returns:
            The deserialized emitted data, formatted as a
            :term:`path timeseries`.
        """
        return path_timeseries_from_data(self.get_data_deserialized(query))

    def get_timeseries(self, query: Optional[list] = None) -> dict:
        """Get the deserialized data as an :term:`embedded timeseries`.

        Returns:
            The deserialized emitted data, formatted as an
            :term:`embedded timeseries`.
        """
        return timeseries_from_data(self.get_data_deserialized(query))


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

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.saved_data: Dict[float, Dict[str, Any]] = {}
        self.fallback_serializer = make_fallback_serializer_function()
        self.embed_path = config.get('embed_path', tuple())

    def emit(self, data: Dict[str, Any]) -> None:
        """
        Emit the timeseries history portion of ``data``, which is
        ``data['data'] if data['table'] == 'history'`` and put it at
        ``data['data']['time']`` in the history.
        """
        if data['table'] == 'history':
            emit_data = data['data'].copy()
            time = emit_data.pop('time', None)
            data_at_time = assoc_path({}, self.embed_path, emit_data)
            self.saved_data.setdefault(time, {})
            data_at_time = serialize_value(
                data_at_time, self.fallback_serializer)
            deep_merge_check(
                self.saved_data[time], data_at_time, check_equality=True)

    def get_data(self, query: Optional[list] = None) -> dict:
        """ Return the accumulated timeseries history of "emitted" data. """
        if query:
            returned_data = {}
            for t, data in self.saved_data.items():
                paths_data = []
                for path in query:
                    datum = get_in(data, path)
                    if datum:
                        path_data = (path, datum)
                        paths_data.append(path_data)
                returned_data[t] = paths_to_dict(paths_data)
            return returned_data
        return self.saved_data


class SharedRamEmitter(RAMEmitter):
    """
    Accumulate the timeseries history portion of the "emitted" data to a table
    in RAM that is shared across all instances of the emitter.
    """

    saved_data: Dict[float, Dict[str, Any]] = {}

    def __init__(self, config: Dict[str, Any]) -> None:  # pylint: disable=super-init-not-called
        # We intentionally don't call the superclass constructor because
        # we don't want to create a per-instance ``saved_data``
        # attribute.
        self.fallback_serializer = make_fallback_serializer_function()
        self.embed_path = config.get('embed_path', tuple())


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
    default_host = 'localhost:27017'
    client_dict: Dict[int, MongoClient] = {}

    @classmethod
    def create_indexes(cls, table: Any, columns: List[Any]) -> None:
        """Create the listed column indexes for the given DB table."""
        for column in columns:
            table.create_index(column)

    def __init__(self, config: Dict[str, Any]) -> None:
        """config may have 'host' and 'database' items."""
        super().__init__(config)
        self.experiment_id = config.get('experiment_id')
        # In the worst case, `breakdown_data` can underestimate the size of
        # data by a factor of 4: len(str(0)) == 1 but 0 is a 4-byte int.
        # Use 4 MB as the breakdown limit to stay under MongoDB's 16 MB limit.
        self.emit_limit = config.get('emit_limit', 4000000)
        self.embed_path = config.get('embed_path', tuple())

        # create new MongoClient per OS process
        curr_pid = os.getpid()
        if curr_pid not in DatabaseEmitter.client_dict:
            DatabaseEmitter.client_dict[curr_pid] = MongoClient(
                config.get('host', self.default_host))
        self.client = DatabaseEmitter.client_dict[curr_pid]

        self.db = getattr(self.client, config.get('database', 'simulations'))
        self.history = getattr(self.db, 'history')
        self.configuration = getattr(self.db, 'configuration')
        self.phylogeny = getattr(self.db, 'phylogeny')
        self.create_indexes(self.history, HISTORY_INDEXES)
        self.create_indexes(self.configuration, CONFIGURATION_INDEXES)
        self.create_indexes(self.phylogeny, CONFIGURATION_INDEXES)

        self.fallback_serializer = make_fallback_serializer_function()

    def emit(self, data: Dict[str, Any]) -> None:
        table_id = data['table']
        table = self.db.get_collection(table_id)
        time = data['data'].pop('time', None)
        data['data'] = assoc_path({}, self.embed_path, data['data'])
        # Analysis scripts expect the time to be at the top level of the
        # dictionary, but some emits, like configuration emits, lack a
        # time key.
        if time is not None:
            data['data']['time'] = time
        emit_data = data.copy()
        emit_data.pop('table', None)
        emit_data['experiment_id'] = self.experiment_id
        self.write_emit(table, emit_data)

    def write_emit(self, table: Any, emit_data: Dict[str, Any]) -> None:
        """Check that data size is less than emit limit.

        Break up large emits into smaller pieces and emit them individually
        """
        assembly_id = str(uuid.uuid4())
        emit_data = serialize_value(emit_data, self.fallback_serializer)
        try:
            emit_data['assembly_id'] = assembly_id
            table.insert_one(emit_data)
        # If document is too large, break up into smaller dictionaries
        # with shared assembly IDs and time keys
        except DocumentTooLarge:
            emit_data.pop('assembly_id')
            experiment_id = emit_data.pop('experiment_id')
            time = emit_data['data'].pop('time', None)
            broken_down_data = breakdown_data(self.emit_limit, emit_data)
            for (path, datum) in broken_down_data:
                d: Dict[str, Any] = {}
                assoc_path(d, path, datum)
                d['assembly_id'] = assembly_id
                d['experiment_id'] = experiment_id
                if time:
                    d.setdefault('data', {})
                    d['data']['time'] = time
                table.insert_one(d)

    def get_data(self, query: Optional[list] = None) -> dict:
        return get_history_data_db(self.history, self.experiment_id, query)


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


def delete_experiment(
    host: str = 'localhost',
    port: Any = 27017,
    query: Optional[dict] = None
) -> None:
    """Helper function to delete experiment data in parallel

    Args:
        host: Host name of database. This can usually be left as the default.
        port: Port number of database. This can usually be left as the
            default.
        query: Filter for documents to delete.
    """
    history_collection = get_local_client(host, port, 'simulations').history
    history_collection.delete_many(query, hint=HISTORY_INDEXES[1])


def delete_experiment_from_database(
    experiment_id: str,
    host: str = 'localhost',
    port: Any = 27017,
    cpus: int = 1
) -> None:
    """Delete an experiment's data from a database.

    Args:
        experiment_id: Identifier of experiment.
        host: Host name of database. This can usually be left as the default.
        port: Port number of database. This can usually be left as the
            default.
        cpus: Number of chunks to split delete operation into to be run in
            parallel. Useful if single-threaded delete does not saturate I/O.
    """
    db = get_local_client(host, port, 'simulations')
    if cpus > 1:
        chunks = get_data_chunks(db.history, experiment_id, cpus=cpus)
        queries = []
        for chunk in chunks:
            queries.append({
                'experiment_id': experiment_id,
                '_id': {'$gte': chunk[0], '$lt': chunk[1]},
                'data.time': {'$gte': MinKey(), '$lte': MaxKey()}
            })
        partial_del_exp = partial(delete_experiment, host, port)
        with ProcessPoolExecutor(cpus) as executor:
            executor.map(partial_del_exp, queries)
    else:
        query = {'experiment_id': experiment_id}
        db.history.delete_many(query, hint=HISTORY_INDEXES[1])
    db.configuration.delete_many(query)


def assemble_data(data: list) -> dict:
    """re-assemble data"""
    assembly: dict = {}
    for datum in data:
        if 'assembly_id' in datum:
            assembly_id = datum['assembly_id']
            if assembly_id not in assembly:
                assembly[assembly_id] = {}
            deep_merge_check(
                assembly[assembly_id],
                datum['data'],
                check_equality=True,
            )
        else:
            assembly_id = str(uuid.uuid4())
            assembly[assembly_id] = datum['data']
    return assembly


def apply_func(
    document: Any,
    field: Tuple,
    f: Optional[Callable[..., Any]] = None,
) -> Any:
    if field[0] not in document:
        return document
    if len(field) != 1:
        document[field[0]] = apply_func(document[field[0]], field[1:], f)
    elif f is not None:
        document[field[0]] = f(document[field[0]])
    return document


def get_query(
    projection: dict,
    host: str,
    port: Any,
    query: dict
) -> list:
    """Helper function for parallel queries

    Args:
        projection: a MongoDB projection in dictionary form
        host, port: used to create new MongoClient for each parallel process
        query: a MongoDB query in dictionary form
    Returns:
        List of projected documents for given query
    """
    history_collection = get_local_client(host, port, 'simulations').history
    return list(history_collection.find(query, projection,
        hint=HISTORY_INDEXES[1]))


def get_data_chunks(
    history_collection: Any,
    experiment_id: str,
    start_time: Union[int, MinKey] = MinKey(),
    end_time: Union[int, MaxKey] = MaxKey(),
    cpus: int = 8
) -> list:
    """Helper function to get chunks for parallel queries

    Args:
        history_collection: the MongoDB history collection to query
        experiment_id: the experiment id which is being retrieved
        start_time, end_time: first and last simulation time to query
        cpus: number of chunks to create
    Returns:
        List of ObjectId tuples that represent chunk boundaries.
        For each tuple, include ``{'_id': {$gte: tuple[0], $lt: tuple[1]}}``
        in the query to search its corresponding chunk.
    """
    id_cutoffs = list(history_collection.aggregate([{
        '$match': {
            'experiment_id': experiment_id,
            'data.time': {'$gte': start_time, '$lte': end_time}}},
        {'$project': {'_id':1}},
        {'$bucketAuto': {'groupBy': '$_id', 'buckets': cpus}},
        {'$group': {'_id': '', 'splitPoints': {'$push': '$_id.min'}}},
        {'$unset': '_id'}],
        hint={'experiment_id':1, 'data.time':1, '_id':1}))[0]['splitPoints']
    id_ranges = []
    for i in range(len(id_cutoffs)-1):
        id_ranges.append((id_cutoffs[i], id_cutoffs[i+1]))
    id_ranges.append((id_cutoffs[-1], MaxKey()))
    return id_ranges


def get_history_data_db(
    history_collection: Any,
    experiment_id: Any,
    query: Optional[list] = None,
    func_dict: Optional[dict[tuple, Callable[..., Any]]] = None,
    f: Optional[Callable[..., Any]] = None,
    filters: Optional[dict] = None,
    start_time: Union[int, MinKey] = MinKey(),
    end_time: Union[int, MaxKey] = MaxKey(),
    cpus: int = 1,
    host: str ='localhost',
    port: Any = '27017'
) -> Dict[float, dict]:
    """Query MongoDB for history data.

    Args:
        history_collection: a MongoDB collection
        experiment_id: the experiment id which is being retrieved
        query: a list of tuples pointing to fields within the experiment data.
            In the format: [('path', 'to', 'field1'), ('path', 'to', 'field2')]
        func_dict: a dict which maps the given query paths to a function that
            operates on the retrieved values and returns the results. If None
            then the raw values are returned.
            In the format: {('path', 'to', 'field1'): function}
        f: a function that applies equally to all fields in query. func_dict
            is the recommended approach and takes priority over f.
        filters: MongoDB query arguments to further filter results
            beyond matching the experiment ID.
        start_time, end_time: first and last simulation time to query
        cpus: splits query into this many chunks to run in parallel, useful if
            single-threaded query does not saturate I/O (e.g. on Google Cloud)
        host: used if cpus>1 to create MongoClient in parallel processes
        port: used if cpus>1 to create MongoClient in parallel processes
    Returns:
        data (dict)
    """

    experiment_query = {'experiment_id': experiment_id}
    if filters:
        experiment_query.update(filters)

    projection = None
    if query:
        projection = {f"data.{'.'.join(field)}": 1 for field in query}
        projection['data.time'] = 1
        projection['assembly_id'] = 1

    if cpus > 1:
        chunks = get_data_chunks(history_collection, experiment_id, cpus=cpus)
        queries = []
        for chunk in chunks:
            queries.append({
                **experiment_query,
                '_id': {'$gte': chunk[0], '$lt': chunk[1]},
                'data.time': {'$gte': start_time, '$lte': end_time}
            })
        partial_get_query = partial(get_query, projection, host, port)
        with ProcessPoolExecutor(cpus) as executor:
            queried_chunks = executor.map(partial_get_query, queries)
        cursor = itertools.chain.from_iterable(queried_chunks)
    else:
        cursor = history_collection.find(experiment_query, projection)
    raw_data = []
    for document in cursor:
        assert document.get('assembly_id'), \
            "all database documents require an assembly_id"
        if ((f is not None) or (func_dict is not None)) and query:
            for field in query:
                if func_dict:  # func_dict takes priority over f
                    func = func_dict.get(field)
                else:
                    func = f

                document["data"] = apply_func(
                    document["data"], field, func)
        raw_data.append(document)

    # re-assemble data
    assembly = assemble_data(raw_data)

    # restructure by time
    data: Dict[float, Any] = {}
    for datum in assembly.values():
        time = datum['time']
        datum = datum.copy()
        datum.pop('_id', None)
        datum.pop('time', None)
        deep_merge_check(
            data,
            {time: datum},
            check_equality=True,
        )

    return data


def get_atlas_client(secrets_path: str) -> Any:
    """Open a MongoDB client using the named secrets config JSON file."""
    with open(secrets_path, 'r') as f:
        secrets = json.load(f)
    emitter_config = get_atlas_database_emitter_config(
        **secrets['database'])
    uri = emitter_config['host']
    client: MongoClient = MongoClient(uri)
    return client[emitter_config['database']]


def get_local_client(host: str, port: Any, database_name: str) -> Any:
    """Open a MongoDB client onto the given host, port, and DB."""
    client: MongoClient = MongoClient('{}:{}'.format(host, port))
    return client[database_name]


def data_from_database(
    experiment_id: str,
    client: Any,
    query: Optional[list] = None,
    func_dict: Optional[dict[tuple, Callable[..., Any]]] = None,
    f: Optional[Callable[..., Any]] = None,
    filters: Optional[dict] = None,
    start_time: Union[int, MinKey] = MinKey(),
    end_time: Union[int, MaxKey] = MaxKey(),
    cpus: int = 1
) -> Tuple[dict, Any]:
    """Fetch something from a MongoDB.

    Args:
        experiment_id: the experiment id which is being retrieved
        client: a MongoClient instance connected to the DB
        query: a list of tuples pointing to fields within the experiment data.
            In the format: [('path', 'to', 'field1'), ('path', 'to', 'field2')]
        func_dict: a dict which maps the given query paths to a function that
            operates on the retrieved values and returns the results. If None
            then the raw values are returned.
            In the format: {('path', 'to', 'field1'): function}
        f: a function that applies equally to all fields in query. func_dict
            is the recommended approach and takes priority over f.
        filters: MongoDB query arguments to further filter results
            beyond matching the experiment ID.
        start_time, end_time: first and last simulation time to query
        cpus: splits query into this many chunks to run in parallel
    Returns:
        data (dict)
    """

    # Retrieve environment config
    config_collection = client.configuration
    experiment_query = {'experiment_id': experiment_id}
    experiment_configs = list(config_collection.find(experiment_query))

    # Re-assemble experiment_config
    experiment_assembly = assemble_data(experiment_configs)
    assert len(experiment_assembly) == 1
    assembly_id = list(experiment_assembly.keys())[0]
    experiment_config = experiment_assembly[assembly_id]

    # Retrieve timepoint data
    history = client.history
    host = client.address[0]
    port = client.address[1]
    data = get_history_data_db(history, experiment_id, query, func_dict,
        f, filters, start_time, end_time, cpus, host, port)

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
