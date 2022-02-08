"""
=====
Store
=====

The file system for storing and updating state variables during an experiment.
"""

import copy
import logging as log
from pprint import pformat
import uuid

import numpy as np
from pint import Quantity
from typing import Optional

from vivarium import divider_registry, serializer_registry, updater_registry
from vivarium.core.process import Process, Step
from vivarium.library.dict_utils import deep_merge, deep_merge_check, MULTI_UPDATE_KEY
from vivarium.library.topology import without, dict_to_paths
from vivarium.core.types import Processes, Topology, State, Steps, Flow

_EMPTY_UPDATES = None, None, None, None, None, None
DEFAULT_DIVIDER = '_default'


def generate_state(
        processes: Processes,
        topology: Topology,
        initial_state: Optional[State],
        steps: Optional[Steps] = None,
        flow: Optional[Flow] = None,
) -> 'Store':
    """Initialize a simulation's state.

    Args:
        processes: Simulation processes.
        topology: Topology linking process ports to stores.
        initial_state: Initial simulation state. Omitted variables will
            be assigned values based on schema defaults.

    Returns:
        Initialized state.
    """
    store = Store({})
    steps = steps or {}
    store.generate(tuple(), processes, steps, flow, topology, initial_state)
    store.build_topology_views()

    return store


def view_values(
        states: dict
) -> State:
    state_values = {}
    if isinstance(states, Store):
        return states.get_value()
    for key, value in states.items():
        state_values[key] = view_values(value)
    return state_values


def key_for_value(d, looking):
    """Get the key associated with a value in a dictionary.

    Only top-level keys are searched.

    Args:
        d: The dictionary.
        looking: The value to look for.

    Returns:
        The associated key, or None if no key found.
    """
    found = None
    for key, value in d.items():
        if looking == value:
            found = key
            break
    return found


def hierarchy_depth(hierarchy, path=()):
    """
    Create a mapping of every path in the hierarchy to the node living at
    that path in the hierarchy.
    """

    base = {}

    for key, inner in hierarchy.items():
        down = tuple(path + (key,))
        if isinstance(inner, dict):
            base.update(hierarchy_depth(inner, down))
        else:
            base[down] = inner

    return base


def _always_true(_):
    return True


def _identity(y):
    return y


def topology_path(topology, path):
    '''
    get the subtopology at the path inside the given topology.
    '''

    if not path:
        topology, path
    else:
        head = path[0]
        tail = path[1:]
        if head in topology:
            subtopology = topology[head]
            if isinstance(subtopology, tuple):
                return subtopology, tail
            elif isinstance(subtopology, dict):
                if '_path' in subtopology:
                    if tail and tail[0] in subtopology:
                        down = topology_path(subtopology, tail)
                        if down:
                            return subtopology['_path'] + down[0], down[1]
                    else:
                        return subtopology['_path'], tail
                else:
                    return topology_path(subtopology, tail)

def insert_topology(topology, port_path, target_path):
    assert isinstance(port_path, tuple)
    assert len(port_path) > 0
    head = port_path[0]
    tail = port_path[1:]

    if not tail:
        topology[head] = target_path
    else:
        subtopology = topology[head]
        if isinstance(subtopology, tuple):
            relative_path = tuple(['..' for _ in subtopology]) + target_path
            new_topology = insert_topology(
                {'_path': subtopology},
                tail,
                relative_path)
            topology[head] = new_topology

        elif isinstance(subtopology, dict):
            topology[head] = insert_topology(subtopology, tail, target_path)

        else:
            raise Exception(f'invalid topology {topology} at key {head}')

    return topology

def convert_path(path):
    if isinstance(path, list):
        path = tuple(path)
    elif not isinstance(path, tuple):
        path = (path,)
    return path

class Store:
    """Holds a subset of the overall model state

    The total state of the model can be broken down into :term:`stores`,
    each of which is represented by an instance of this `Store` class.
    The store's state is a set of :term:`variables`, each of which is
    defined by a set of :term:`schema key-value pairs`. The valid schema
    keys are listed in :py:attr:`schema_keys`, and they are:

    * **_default** (Type should match the variable value): The default
      value of the variable.
    * **_updater** (:py:class:`str`): The name of the :term:`updater` to
      use. By default this is ``accumulate``.
    * **_divider** (:py:class:`str`): The name of the :term:`divider` to
      use. Note that ``_divider`` is not included in the ``schema_keys``
      set because it can be applied to any node in the hierarchy, not
      just leaves (which represent variables).
    * **_value** (Type should match the variable value): The current
      value of the variable. This is ``None`` by default.
    * **_properties** (:py:class:`dict`): Extra properties of the
      variable that don't have a specific schema key. This is an empty
      dictionary by default.
    * **_emit** (:py:class:`bool`): Whether to emit the variable to the
      :term:`emitter`. This is ``False`` by default.
    """
    schema_keys = {
        '_default',
        '_updater',
        '_value',
        '_properties',
        '_emit',
        '_serializer',
    }

    def __init__(self, config, outer=None, source=None):
        self.outer = outer
        self.inner = {}
        self.subschema = {}
        self.subtopology = {}
        self.properties = {}
        self.default = None
        self.updater = None
        self.value = None
        self.units = None
        self.divider = None
        self.emit = False
        self.sources = {}
        self.leaf = False
        self.serializer = None
        self.topology = {}
        self.topology_view = None
        # self.flow is None when this node has no flow (either because
        # it is not a Step or because it is a Step treated like a
        # Deriver) and a list when this node holds a Step with
        # dependencies. The list is empty if the Step has no
        # dependencies but should not be treated like a Deriver.
        self.flow = None

        self._apply_config(config, source)

    def __getitem__(self, path):
        '''Retrieve a :term:`hierarchy` node by its :term:`path`.

        .. WARNING:

           This function is **experimental** and part of the
           :term:`store API`.

        Args:
            path: Path, relative to ``self``, of the store to retrieve.

        Returns:
            The store at ``path``. Note that the store is returned, not
            the value of that store.
        '''
        path = convert_path(path)
        return self.get_path(path)

    def __setitem__(self, path, value):
        '''Set a :term:`hierarchy` node's value by its :term:`path`.

        .. WARNING:

           This function is **experimental** and part of the
           :term:`store API`.

        Args:
            path: Path, relative to ``self``, of the store to modify.
            value: The value to be stored as the value of the store at
                ``path``.
        '''
        path = convert_path(path)
        self.set_path(path, value)

    def create(self, path, value=None, absolute=False, **kwargs):
        path = convert_path(path)
        if value:
            kwargs['_value'] = value
        if '_default' not in kwargs:
            kwargs['_default'] = kwargs.get('_value')

        if absolute:
            top = self.top()
            store = top._establish_path(path, config=kwargs)
            top._apply_subschema_path(path)
        else:
            store = self._establish_path(path, config=kwargs)
            self._apply_subschema_path(path)

        store.apply_defaults()
        return store

    def connect(self, path, value, absolute=False):
        '''Wire a store's process to another store.

        This function must not be used unless ``self`` holds a
        :term:`process`.

        .. WARNING:

           This function is **experimental** and part of the
           :term:`store API`.

        Args:
            path: Path of the port to connect.
            value: The store (or the path to the store) to connect to
                the port at ``path``.

        Raises:
            AssertionError: If ``self.value`` is not an instance of
                :py:class:`vivarium.core.process.Process`.
            Exception: If ``value`` is a :py:class:`Store` that is in a
                different tree than ``self``.
        '''
        path = convert_path(path)
        assert isinstance(self.value, Process), \
            f'cannot connect non-process {self.value} at {self.path_for()} to {path}'

        if isinstance(value, Store):
            target_store = value
            if self._independent_store(target_store):
                raise Exception(
                    f"the store being inserted at {path} is from a different tree "
                    f"at {target_store.path_for()}: {target_store.get_value()}")
        else:
            store_path = convert_path(value)
            if absolute:
                target_store = self.top().get_path(store_path)
            else:
                target_store = self.outer.get_path(store_path)

        # update the topology
        self._update_topology(path, target_store)

    def set_path(self, path, value):
        '''Set a value at a path in the hierarchy.

        .. WARNING:

           This function is **experimental** and part of the
           :term:`store API`.

        Args:
            path: The :term:`path` relative to ``self`` where the value
                should be set.
            value: The value to set. The store node at ``path`` will
                hold ``value`` when this function returns.
        '''

        # this case only when called directly
        if len(path) == 0:
            if isinstance(value, Store):
                self.value = value.value
            else:
                self.value = value

        elif len(path) == 1:
            final = path[0]
            if isinstance(value, Store):
                raise Exception(
                    f"the store being inserted at {path} is already in a tree "
                    f"at {value.path_for()}: {value.get_value()}")

                # TODO: make it so a Store can be added here if it has no outer
                # if value.outer:
                #     raise Exception(f"the store being inserted at {path} is already in a tree
                #     at {value.path_for()}: {value.get_value()}")
                # else:
                #     # place the store at this point in the tree
                #     self.inner[final] = value
                #     value.outer = self

            elif isinstance(value, Process):
                self.insert({
                    'processes': value.generate_processes({'name': final}),
                    'topology': value.generate_topology({'name': final}),
                    'initial_state': {},
                })

            else:
                down = self.get_path((final,))
                if down:
                    if not down.leaf:
                        Exception(f'trying to set the value {value} of a branch at {down.path_for()}')
                    down.value = value
                else:
                    Exception(f'trying to set the value {value} at a path that does not exist {final} at {self.path_for()}')

        elif len(path) > 1:
            head = path[0]
            tail = path[1:]
            down = self.get_path((head,))
            if down:
                down.set_path(tail, value)
            else:
                Exception(f'trying to set the value {value} at a path that does not exist {path} at {self.path_for()}')

        else:
            raise Exception("this should never happen")

    def path_to(self, to):
        """return a path from self to the given Store"""

        self_path = self.path_for()
        to_path = to.path_for()
        while len(self_path) > 0 and len(to_path) > 0 and self_path[0] == to_path[0]:
            self_path = self_path[1:]
            to_path = to_path[1:]

        path = [
            '..'
            for _ in self_path]
        path.extend(to_path)
        return tuple(path)

    def _update_topology(self, port_path, target_store):
        """Update the topology with a new port-path pair.

        To use this function ``self`` must hold a :term:`process`.

        .. WARNING:

           This function is **experimental** and part of the
           :term:`store API`.

        Args:
            port_path: Port of the new topology entry.
            target_store: The store to wire to ``port_path``.

        Raises:
            AssertionError: If ``self.value`` is not an instance of
                :py:class:`vivarium.core.process.Process`.
        """

        assert isinstance(self.value, Process), \
            f'assigning topology from {port_path} to {target_store.path_for()} ' \
            f'at {self.path_for()} is invalid, not a process'

        topology = copy.deepcopy(self.topology)
        self.topology = insert_topology(
            topology, port_path, self.outer.path_to(target_store))

        self.value.schema = self.value.get_schema()
        self.outer._topology_ports(
            self.value.schema,
            self.topology,
            source=self.path_for())

        # cache the process's view
        self.topology_view = self.schema_topology(
            self.value.schema,
            self.topology)

    def _independent_store(self, store):
        return self.top() != store.top()

    def _check_default(self, new_default):
        """Check a new default value.

        Compare a new default value to the existing default. If they
        conflict, decide which to rely on and log a warning.

        Returns:
            The new default value.
        """
        defaults_conflict = False
        if self.default is not None:
            self_default_comp = self.default
            new_default_comp = new_default
            if isinstance(self_default_comp, np.ndarray):
                self_default_comp = self.default.tolist()
            if isinstance(new_default_comp, np.ndarray):
                new_default_comp = new_default.tolist()
            defaults_conflict != (self_default_comp == new_default_comp)
        if defaults_conflict:
            if (
                not isinstance(new_default, np.ndarray)
                and not isinstance(self.default, np.ndarray)
                and new_default == 0
                and self.default != 0
            ):
                log.debug(
                    '_default schema conflict: %s and %s. selecting %s',
                    str(self.default), str(new_default), str(self.default))
                return self.default
            log.debug(
                '_default schema conflict: %s and %s. selecting %s',
                str(self.default), str(new_default), str(new_default))
        return new_default

    def _check_value(self, new_value):
        """Check a new schema value.

        Args:
            new_value: The new value.

        Returns:
            The new value.

        Raises:
            Exception: If the store already has a value and the new
                value is different from the existing one.
        """
        if self.value is not None and new_value != self.value:
            raise Exception(
                '_value schema conflict: {} and {}'.format(
                    new_value, self.value))
        return new_value

    def _merge_subtopology(self, subtopology):
        """Merge a new subtopology with the store's existing one."""
        self.subtopology = deep_merge(self.subtopology, subtopology)

    def _apply_subschema_config(self, subschema):
        """Merge a new subschema config with the current subschema."""
        self.subschema = deep_merge(
            self.subschema,
            subschema)

    def _apply_config(self, config, source=None):
        """
        Expand the tree by applying additional config.

        Special keys for the config are:

        * _default - Default value for this node.
        * _properties - An arbitrary map of keys to values. This can be used
          for any properties which exist outside of the operation of the
          tree (like mass or energy).
        * _updater - Which updater to use. Default is 'accumulate' which
          adds the new value to the existing value, but 'set' is common
          as well. You can also provide your own function here instead
          of a string key into the updater library.
        * _emit - whether or not to emit the values under this point in
           the tree.
        * _divider - What to do with this node when division happens.
          Default behavior is to leave it alone, but you can also pass
          'split' here, or a function of your choosing. If you need
          other values from the state you need to supply a dictionary
          here containing the updater and the topology for where the
          other state values are coming from. This has two keys:

            * divider - a function that takes the existing value and any
              values supplied from the adjoining topology.
            * topology - a mapping of keys to paths where the value for
              those keys will be found. This will be passed in as the
              second argument to the divider function.

        * _subschema/* - If this node was declared to house an unbounded set
          of related states, the schema for these states is held in this
          nodes subschema and applied whenever new subkeys are added
          here.
        * _subtopology - The subschema is informed by the subtopology to
          map the process perspective to the actual tree structure.
        * _topology - If this node stores a :term:`process`, then the
          process's topology must be provided under this key. This key
          may only be provided if the node stores a :term:`process`.
        * _flow - If this node stores a :term:`step`, then the step's
          dependencies must be specified under this key as a list of
          :term:`paths` relative to the step's parent node in the
          :term:`hierarchy`. This key must not be provided unless the
          node holds a step.
        """

        if config == '**':
            config = {}  # config needs to be a dict

        # remove _output special key. This is used only by schema_topology.
        config = without(config, '_output')

        if '*' in config:
            self._apply_subschema_config(config['*'])
            config = without(config, '*')

        if '_subschema' in config:
            if source:
                self.sources[source] = config['_subschema']
            self._apply_subschema_config(config['_subschema'])
            config = without(config, '_subschema')

        if '_subtopology' in config:
            self._merge_subtopology(config['_subtopology'])
            config = without(config, '_subtopology')

        if '_topology' in config:
            self.topology = config['_topology']
            config = without(config, '_topology')

        if '_flow' in config:
            flow = config.pop('_flow')
            if flow != {}:
                self.flow = flow

        if '_divider' in config:
            new_divider = config['_divider']
            if isinstance(new_divider, str):
                new_divider = divider_registry.access(new_divider)
            if isinstance(new_divider, dict) and isinstance(
                    new_divider['divider'], str):
                new_divider['divider'] = divider_registry.access(
                    new_divider['divider'])
            if (
                    self.divider
                    and self.divider != DEFAULT_DIVIDER
                    and self.divider != new_divider):
                raise ValueError(
                    f'Trying to assign divider {new_divider} to '
                    f'{self.path_for()}, which already has divider '
                    f'{self.divider}.')
            self.divider = new_divider
            config = without(config, '_divider')

        if self.schema_keys & set(config.keys()):
            if self.inner:
                raise Exception(
                    'trying to assign leaf values to a branch at: {}'.format(
                        self.path_for()))
            self.leaf = True

            if '_units' in config:
                self.units = config['_units']
                self.serializer = serializer_registry.access('units')

            if '_serializer' in config:
                self.serializer = config['_serializer']
                if isinstance(self.serializer, str):
                    self.serializer = serializer_registry.access(
                        self.serializer)

            if '_default' in config:
                self.default = self._check_default(config.get('_default'))
                if isinstance(self.default, Quantity):
                    self.units = self.units or self.default.units
                    self.serializer = (self.serializer or
                                       serializer_registry.access('units'))
                elif isinstance(self.default, list) and \
                        len(self.default) > 0 and \
                        isinstance(self.default[0], Quantity):
                    self.units = self.units or self.default[0].units
                    self.serializer = (self.serializer or
                                       serializer_registry.access('units'))
                elif isinstance(self.default, np.ndarray):
                    self.serializer = (self.serializer or
                                       serializer_registry.access('numpy'))

            if '_value' in config:
                self.value = self._check_value(config.get('_value'))
                if isinstance(self.value, Quantity):
                    self.units = self.value.units

            self.updater = config.get(
                '_updater',
                self.updater or 'accumulate',
            )

            # All leaf nodes must have a divider, even though a divider
            # on a branch node higher in the tree will take precedence.
            self.divider = self.divider or DEFAULT_DIVIDER

            self.properties = deep_merge(
                self.properties,
                config.get('_properties', {}))

            self.emit = config.get('_emit', self.emit)

            if source:
                self.sources[source] = config

        else:
            if self.leaf and config:
                if self.value:
                    raise Exception(
                        f'trying to assign create inner for leaf node: '
                        f'{self.path_for()} with value {self.value}')
                self.leaf = False

            for key, child in config.items():
                if key not in self.inner:
                    self.inner[key] = Store(child, outer=self, source=source)
                else:
                    self.inner[key]._apply_config(child, source=source)

        if self.topology and not isinstance(self.value, Process):
            raise ValueError(
                f'Attempting to create Store at {self.path_for()} '
                f'with topology {self.topology}, which is not allowed '
                f'because the Store value ({self.value}) is not a '
                'Process.')
        if self.flow and not self.value.is_step():
            raise ValueError(
                f'Attempting to create Store at {self.path_for()} '
                f'with flow {self.flow}, which is not allowed because '
                f'the Store value ({self.value}) is not a Step.')

    def _get_updater(self, update):
        """Get the updater to use for an update applied to this store.

        Args:
            update: The update.

        Returns:
            If available, the updater specified in the update. If no
            such updater is specified, return this store's default
            updater. If necessary, retrieves updater from the registry.
        """
        updater = self.updater
        if isinstance(update, dict) and '_updater' in update:
            updater = update['_updater']

        if isinstance(updater, str):
            updater = updater_registry.access(updater)
        return updater

    def _get_divider(self):
        if self.divider == DEFAULT_DIVIDER:
            if self.topology:
                # For processes, we use a null divider by default.
                return divider_registry.access('null')
            # For all other nodes, by default we use a 'set' divider.
            return divider_registry.access('set')
        return self.divider

    def get_config(self, sources=False):
        """
        Assemble a dictionary representation of the config for this node.
        A desired property is that the node can be exactly recreated by
        applying the resulting config to an empty node again.
        """

        config = {}

        if self.properties:
            config['_properties'] = self.properties
        if self.subschema:
            config['_subschema'] = self.subschema
        if self.subtopology:
            config['_subtopology'] = self.subtopology
        if self.divider and self.divider != DEFAULT_DIVIDER:
            config['_divider'] = self.divider

        if sources and self.sources:
            config['_sources'] = self.sources

        if self.inner:
            child_config = {
                key: child.get_config(sources)
                for key, child in self.inner.items()}
            config.update(child_config)

        else:
            config.update({
                '_default': self.default,
                '_value': self.value})
            if self.updater:
                config['_updater'] = self.updater
            if self.units:
                config['_units'] = self.units
            if self.emit:
                config['_emit'] = self.emit

        return config

    def top(self):
        """
        Find the top of this tree.
        """

        if self.outer:
            return self.outer.top()
        return self

    def path_for(self):
        """
        Find the path to this node.
        """

        if self.outer:
            key = key_for_value(self.outer.inner, self)
            above = self.outer.path_for()
            return above + (key,)
        return tuple()

    def get_value(self, condition=None, f=None):
        """
        Pull the values out of the tree in a structure symmetrical to the tree.
        """

        if self.inner:
            if condition is None:
                condition = _always_true

            if f is None:
                f = _identity

            return {
                key: f(child.get_value(condition, f))
                for key, child in self.inner.items()
                if condition(child)}
        if self.subschema:
            return {}
        elif self.topology:
            # this is a process, return it with the topology
            return (self.value, self.topology)
        return self.value

    def get_processes(self):
        """
        Get all processes in this store. Does not include steps.
        """

        if self.inner:
            inner_processes = {}
            for key, child in self.inner.items():
                if child.inner:
                    child_processes = child.get_processes()
                    if child_processes:
                        inner_processes[key] = child_processes
                elif (
                        isinstance(child.value, Process)
                        and not child.value.is_step()):
                    inner_processes[key] = child.value
            if inner_processes:
                return inner_processes
        elif isinstance(self.value, Process):
            return self.value
        return None

    def get_steps(self):
        """Get all steps under this store."""
        if self.inner:
            inner_processes = {}
            for key, child in self.inner.items():
                if child.inner:
                    child_processes = child.get_steps()
                    if child_processes:
                        inner_processes[key] = child_processes
                elif (
                        isinstance(child.value, Process)
                        and child.value.is_step()):
                    inner_processes[key] = child.value
            if inner_processes:
                return inner_processes
        elif isinstance(self.value, Process):
            return self.value
        return None

    def get_topology(self):
        """
        Get the topology for all processes in this store.
        """

        if self.inner:
            inner_topology = {}
            for key, child in self.inner.items():
                child_topology = child.get_topology()
                if child_topology:
                    inner_topology[key] = child_topology
            if inner_topology:
                return inner_topology
        elif self.topology:
            return self.topology
        return None

    def get_flow(self):
        """Get the flow for all :term:`steps` under this node.

        For example:

        >>> from vivarium.core.store import Store
        >>> from vivarium.core.process import Step
        >>> class MyStep(Step):
        ...     def ports_schema(self):
        ...         return {
        ...             'port': ['variable'],
        ...         }
        ...     def next_update(self, timestep, states):
        ...         return {}
        >>> schema = {
        ...     'agent1': {
        ...         'store': {
        ...             'variable': {
        ...                 '_default': 0,
        ...             },
        ...         },
        ...         'step1': {
        ...             '_value': MyStep(),
        ...             '_topology': {
        ...                 'port': ('store',),
        ...             },
        ...             '_flow': [],
        ...         },
        ...         'step2': {
        ...             '_value': MyStep(),
        ...             '_topology': {
        ...                 'port': ('store',),
        ...             },
        ...             '_flow': [('step1',)],
        ...         },
        ...     },
        ... }
        >>> store = Store(schema)
        >>> store.get_flow()
        {'agent1': {'step1': [], 'step2': [('step1',)]}}
        """
        if self.inner:
            inner_flow = {}
            for key, child in self.inner.items():
                child_flow = child.get_flow()
                if child_flow is not None:
                    inner_flow[key] = child_flow
            if inner_flow:
                return inner_flow
        elif self.flow is not None:
            return self.flow
        return None

    def get_path(self, path):
        """
        Get the node at the given path relative to this node.
        """

        if path:
            step = path[0]
            if step == '..':
                child = self.outer
            else:
                child = self.inner.get(step)

            if child:
                return child.get_path(path[1:])
            elif isinstance(self.value, Process):
                towards = topology_path(self.topology, path)
                if towards:
                    target = self.outer.get_path(towards[0])
                    return target.get_path(towards[1])
            else:
                raise Exception(
                    f'{path} is not a valid path from '
                    f'{self.path_for()}')
        return self

    def get_paths(self, paths):
        """Get the nodes at each of the specified paths.

        Args:
            paths: Map from keys to paths.

        Returns:
            A dictionary with the same keys as ``paths``. Each key is
            mapped to the Store object at the associated path.
        """
        return {
            key: self.get_path(path)
            for key, path in paths.items()}

    def get_values(self, paths):
        """Get the values at each of the provided paths.

        Args:
            paths: Map from keys to paths.

        Returns:
            A dictionary with the same keys as ``paths``. Each key is
            mapped to the value at the associated path.
        """
        return {
            key: self.get_in(path)
            for key, path in paths.items()}

    def get_in(self, path):
        """Get the value at ``path`` relative to this store."""
        return self.get_path(path).get_value()

    def get_template(self, template):
        """
        Pass in a template dict with None for each value you want to
        retrieve from the tree!
        """

        state = {}
        for key, value in template.items():
            child = self.inner[key]
            if value is None:
                state[key] = child.get_value()
            else:
                state[key] = child.get_template(value)
        return state

    def emit_data(self):
        """Emit the value at this Store.

        Obeys the schema (namely emits only if ``_emit`` is true). Also
        applies serializers and converts units as necessary.

        Returns:
            The value to emit, or None if nothing should be emitted.
        """
        data = {}
        if self.inner:
            for key, child in self.inner.items():
                child_data = child.emit_data()
                if child_data is not None or child_data == 0:
                    data[key] = child_data
            return data
        if self.emit:
            if self.serializer:
                if isinstance(self.value, list) and self.units:
                    return self.serializer.serialize(
                        [v.to(self.units) for v in self.value])
                if self.units:
                    return self.serializer.serialize(
                        self.value.to(self.units))
                if isinstance(self.value, list):
                    return self.value
                return self.serializer.serialize(self.value)
            if self.units:
                return self.value.to(self.units).magnitude
            return self.value
        return None

    def _delete_path(self, path):
        """
        Delete the subtree at the given path.
        """

        if not path:
            self.inner = {}
            self.value = None
            return self
        target = self.get_path(path[:-1])
        remove = path[-1]
        if remove in target.inner:
            lost = target.inner[remove]
            del target.inner[remove]
            return lost
        return None

    def divide_value(self):
        """
        Apply the divider for each node to the value in that node to
        assemble two parallel divided states of this subtree.
        """
        divider = self._get_divider()
        if divider:
            # divider is either a function or a dict with topology and/or config
            if isinstance(divider, dict):
                divider_dict = divider
                divider = divider['divider']
                if isinstance(divider, str):
                    divider = divider_registry.access(divider)
                args = {}
                if 'topology' in divider_dict:
                    topology = divider_dict['topology']
                    args.update({'state': self.topology_state(topology)})
                if 'config' in divider_dict:
                    config = divider_dict['config']
                    args.update({'config': config})

                return divider(self.get_value(), **args)
            return divider(self.get_value())
        if self.inner:
            daughters = [{}, {}]
            for key, child in self.inner.items():
                division = child.divide_value()
                if division:
                    for daughter, divide in zip(daughters, division):
                        daughter[key] = divide
            return daughters
        return None

    def _reduce(self, reducer, initial=None):
        """
        Call the reducer on each node accumulating over the result.
        """

        value = initial

        for path, node in self.depth():
            value = reducer(value, path, node)
        return value

    def set_value(self, value):
        """
        Set the value for the given tree elements directly instead of using
        the updaters from their nodes.
        """

        if self.inner or self.subschema:
            if not isinstance(value, dict):
                raise Exception(f"trying to set branch {self.path_for()} to value {value}")

            for child, inner_value in value.items():
                if child not in self.inner:
                    if self.subschema:
                        self.inner[child] = Store(self.subschema, self)
                    else:
                        pass
                        # TODO: continue to ignore extra keys?

                if child in self.inner:
                    self.inner[child].set_value(inner_value)
        else:
            self.value = value

    def generate_value(self, value):
        """
        generate the structure for this value that don't exist,
        but don't overwrite any existing values.
        """

        if self.inner or self.subschema:
            if not isinstance(value, dict):
                raise Exception(f"trying to set branch {self.path_for()} to value {value}")

            for child, inner_value in value.items():
                if child not in self.inner:
                    if self.subschema:
                        self.inner[child] = Store(self.subschema, self)
                    else:
                        self._establish_path((child,), {})

                if child in self.inner:
                    self.inner[child].generate_value(inner_value)
        else:
            if self.value is None:
                self.value = value

    def apply_defaults(self):
        """
        If value is None, set to default.
        """
        if self.inner:
            for child in self.inner.values():
                child.apply_defaults()
        else:
            if self.value is None:
                self.value = self.default

    def add(self, added):
        key = added['key']
        inner_keys = self.inner.keys()
        if key in inner_keys:
            raise Exception(
                f"cannot add '{key}' to the hierarchy, "
                f"already present at path {self.path_for()} ")
        elif key == '_unique_id':
            path = (str(uuid.uuid1()),)
        else:
            path = (key,)
        added_state = added['state']

        # get path
        target = self._establish_path(path, {})
        self._apply_subschema_path(path)
        target.apply_defaults()
        target.set_value(added_state)

    def move(self, move, process_store):
        '''
        .. WARNING:

           While this function is used internally by :py:class:`Store`,
           using it as a public method is **experimental** and part of
           the :term:`store API`.
        '''
        process_updates = []
        step_updates = []
        flow_updates = []
        topology_updates = []
        deletions = []

        # get the source node
        source_key = move['source']
        source_path = (source_key,)
        source_node = self.get_path(source_path)

        # move source node to target path
        target_port = move['target']
        target_topology = process_store.topology[target_port]
        target_node = process_store.outer.get_path(target_topology)
        target = target_node.add_node(source_path, source_node)
        target_path = target.path_for() + source_path

        # find the paths to all the processes
        source_process_paths = source_node.depth(
            filter_function=lambda x: isinstance(x.value, Process))

        # find the process and topology updates
        for path, process in source_process_paths:
            process_path = target_path + path
            process_updates.append((
                process_path, process.value))
            topology_updates.append((
                process_path, process.topology))
            if process.value.is_step():
                step_updates.append((
                    process_path, process.value))
                # Note that process.flow may be None, indicating no
                # flow.
                flow_updates.append((
                    process_path, process.flow))

        self._delete_path(source_path)

        here = self.path_for()
        source_absolute = tuple(here + source_path)
        deletions.append(source_absolute)

        return (
            process_updates, step_updates, flow_updates,
            topology_updates, deletions)

    def insert(self, insertion):
        '''
        .. WARNING:

           While this function is used internally by :py:class:`Store`,
           using it as a public method is **experimental** and part of
           the :term:`store API`.
        '''
        process_updates = []
        step_updates = []
        flow_updates = []
        topology_updates = []

        key = insertion.get('key')
        path = (key,) if key else tuple()
        here = self.path_for()

        self.generate(
            path,
            insertion['processes'],
            insertion.get('steps', {}),
            insertion.get('flow'),
            insertion['topology'],
            insertion['initial_state'])

        root = here + path
        process_paths = dict_to_paths(root, insertion['processes'])
        process_updates.extend(process_paths)

        step_paths = dict_to_paths(root, insertion.get('steps', {}))
        step_updates.extend(step_paths)

        topology_paths = [
            (root + (key,), topology)
            for key, topology in insertion['topology'].items()]
        topology_updates.extend(topology_paths)

        flow_paths = [
            (root + (key,), flow)
            for key, flow in insertion.get('flow', {}).items()]
        flow_updates.extend(flow_paths)

        self._apply_subschema_path(path)
        self.get_path(path).apply_defaults()

        return process_updates, step_updates, flow_updates, topology_updates

    def divide(self, divide):
        '''
        .. WARNING:

           While this function is used internally by :py:class:`Store`,
           using it as a public method is **experimental** and part of
           the :term:`store API`.
        '''
        process_and_step_updates = []
        process_updates = []
        step_updates = []
        flow_updates = []
        topology_updates = []
        deletions = []

        # use dividers to find initial states for daughters
        mother = divide['mother']
        mother_path = (mother,)
        daughters = divide['daughters']
        daughter_states = self.inner[mother].divide_value()

        here = self.path_for()

        for daughter, daughter_state in \
                zip(daughters, daughter_states):
            # use initial state as default, merge in divided values
            merged_initial_state = deep_merge(
                daughter_state, daughter.get('initial_state', {}))

            daughter_key = daughter['key']
            daughter_path = (daughter_key,)

            # get the daughter processes
            if 'processes' in daughter or 'steps' in daughter:
                processes = daughter['processes']
                deep_merge_check(processes, daughter.get('steps', {}))
            else:
                # if no processes provided, copy the mother's processes
                mother_processes = self.get_path(mother_path).get_processes()
                processes = copy.deepcopy(mother_processes)
                processes = processes or {}

            # get the daughter topology
            if 'topology' in daughter:
                topology = daughter['topology']
            else:
                # if no topology provided, copy the mother's topology
                mother_topology = self.get_path(mother_path).get_topology()
                topology = copy.deepcopy(mother_topology)
                topology = topology or {}

            # get the daughter flow
            if 'flow' in daughter:
                flow = daughter['flow']
            else:
                # if no flow provided, copy the mother's flow
                mother_flow = self.get_path(mother_path).get_flow()
                flow = copy.deepcopy(mother_flow)
                flow = flow or {}

            self.generate(
                daughter_path,
                processes,
                {},
                flow,
                topology,
                merged_initial_state)

            root = here + daughter_path
            process_paths = dict_to_paths(root, processes)
            process_and_step_updates.extend(process_paths)

            flow_paths = dict_to_paths(root, flow)
            flow_updates.extend(flow_paths)

            topology_paths = [
                (root + (key,), ports)
                for key, ports in topology.items()]
            topology_updates.extend(topology_paths)

            self._apply_subschema_path(daughter_path)
            target = self.get_path(daughter_path)
            target.apply_defaults()
            target.set_value(merged_initial_state)


        self._delete_path(mother_path)
        deletions.append(tuple(here + mother_path))

        for path, process in process_and_step_updates:
            if process.is_step():
                step_updates.append((path, process))
            else:
                process_updates.append((path, process))

        return (
            process_updates, step_updates, flow_updates,
            topology_updates, deletions)

    def delete(self, key, here=None):
        '''
        .. WARNING:

           While this function is used internally by :py:class:`Store`,
           using it as a public method is **experimental** and part of
           the :term:`store API`.
        '''
        if here is None:
            here = self.path_for()
        deletions = []
        path = (key,)
        self._delete_path(path)
        deletions.append(tuple(here + path))

        return deletions

    def apply_update(self, update, state=None):
        """
        Given an arbitrary update, map all the values in that update
        to their positions in the tree where they apply, and update
        these values using each node's `_updater`.

        Arguments:
            update: The update being applied.
            state: The state at the start of the time step.

        There are five topology update methods, which use the following
        special update keys:

        * `_add` - Adds states into the subtree, given a list of dicts
            containing:

            * path - Path to the added state key.
            * state - The value of the added state.

        * `_move` - Moves a node from a source to a target location in the
          tree. This uses an update to an :term:`outer` port, which
          contains both the source and target node locations. Can move
          multiple nodes according to a list of dicts containing:

            * source - the source path from an outer process port
            * target - the location where the node will be placed.

        * `_generate` - The value has four keys, which are essentially
          the arguments to the `generate()` function:

            * path - Path into the tree to generate this subtree.
            * processes - Tree of processes to generate.
            * topology - Connections of all the process's `ports_schema()`.
            * initial_state - Initial state for this new subtree.

        * `_divide` - Performs cell division by constructing two new
          daughter cells and removing the mother. Takes a dict with two keys:

            * mother - The id of the mother (for removal)
            * daughters - List of two new daughter generate directives, of the
                same form as the `_generate` value above.

        * `_delete` - The value here is a list of paths (tuples) to delete from
          the tree.


        Additional special update keys for different update operations:

        * `_updater` - Override the default updater with any updater you want.

        * `_reduce` - This allows a reduction over the entire subtree from some
          point downward. Its three keys are:

            * from - What point to start the reduction.
            * initial - The initial value of the reduction.
            * reducer - A function of three arguments, which is called
              on every node from the `from` point in the tree down:

                * value - The current accumulated value of the reduction.
                * path - The path to this point in the tree
                * node - The actual node being visited.

              This function returns the next `value` for the reduction.
              The result of the reduction will be assigned to this point
              in the tree.
        """
        view_expire = False

        if isinstance(update, dict) and MULTI_UPDATE_KEY in update:
            # apply multiple updates to same node
            multi_update = update[MULTI_UPDATE_KEY]
            assert isinstance(multi_update, list)
            for update_value in multi_update:
                self.apply_update(update_value, state)
            return _EMPTY_UPDATES

        if self.inner or self.subschema:
            # Branch update: this node has an inner
            process_updates = []
            step_updates = []
            flow_updates = []
            topology_updates = []
            deletions = []

            update = dict(update)  # avoid mutating the caller's dict

            add_entries = update.pop('_add', None)
            if add_entries is not None:
                # add a list of sub-states
                for added in add_entries:
                    self.add(added)
                view_expire = True

            move_entries = update.pop('_move', None)
            if move_entries is not None:
                # move nodes from source to target path
                for move in move_entries:
                    (
                        move_processes, move_step, move_flow,
                        move_topology, move_deletions
                    ) = self.move(move, state)
                    process_updates.extend(move_processes)
                    step_updates.extend(move_step)
                    flow_updates.extend(move_flow)
                    topology_updates.extend(move_topology)
                    deletions.extend(move_deletions)
                    view_expire = True

            generate_entries = update.pop('_generate', None)
            if generate_entries is not None:
                # generate a list of new processes
                for generate in generate_entries:
                    (
                        insert_processes, insert_steps, insert_flows,
                        insert_topology
                    ) = self.insert(generate)
                    process_updates.extend(insert_processes)
                    step_updates.extend(insert_steps)
                    flow_updates.extend(insert_flows)
                    topology_updates.extend(insert_topology)
                    view_expire = True

            divide = update.pop('_divide', None)
            if divide is not None:
                (
                    divide_processes, divide_steps, divide_flow,
                    divide_topology, divide_deletions
                ) = self.divide(divide)
                process_updates.extend(divide_processes)
                step_updates.extend(divide_steps)
                flow_updates.extend(divide_flow)
                topology_updates.extend(divide_topology)
                deletions.extend(divide_deletions)
                view_expire = True

            delete_keys = update.pop('_delete', None)

            for key, value in update.items():
                if key in self.inner:
                    inner = self.inner[key]
                    (
                        inner_topology, inner_processes, inner_steps,
                        inner_flows, inner_deletions, inner_view_expire
                    ) = inner.apply_update(value, state)

                    if inner_topology:
                        topology_updates.extend(inner_topology)
                    if inner_processes:
                        process_updates.extend(inner_processes)
                    if inner_steps:
                        step_updates.extend(inner_steps)
                    if inner_flows:
                        flow_updates.extend(inner_flows)
                    if inner_deletions:
                        deletions.extend(inner_deletions)
                    if inner_view_expire:
                        view_expire = inner_view_expire

            if delete_keys is not None:
                # delete a list of paths
                here = self.path_for()
                for key in delete_keys:
                    delete_deletions = self.delete(key, here)
                    deletions.extend(delete_deletions)
                view_expire = True

            return (
                topology_updates, process_updates, step_updates,
                flow_updates, deletions, view_expire)

        # Leaf update: this node has no inner

        updater = self._get_updater(update)

        if isinstance(update, dict) and '_reduce' in update:
            reduction = update['_reduce']
            top = self.get_path(reduction.get('from'))
            update = top._reduce(
                reduction['reducer'],
                initial=reduction['initial'])

        if isinstance(update, dict) and \
                self.schema_keys & set(update.keys()):
            if '_updater' in update:
                update = update.get('_value', self.default)

        try:
            self.value = updater(self.value, update)
        except:
            # provide error messages
            if updater is None:
                raise Exception(
                    f"updater is absent at path {self.path_for()} "
                    f"with value {self.value} for update {pformat(update)}")
            else:
                raise Exception(
                    f"failed update at path {self.path_for()} "
                    f"with value {self.value} for update {pformat(update)}")

        return _EMPTY_UPDATES

    def inner_value(self, key):
        """
        Get the value of an inner state
        """

        if key in self.inner:
            return self.inner[key].get_value()
        return None

    def topology_state(self, topology):
        """
        Fill in the structure of the given topology with the values at all
        the paths the topology points at. Essentially, anywhere in the topology
        that has a tuple path will be filled in with the value at that path.

        This is the inverse function of the standalone `inverse_topology`.
        """

        state = {}

        for key, path in topology.items():
            if key == '*':
                if isinstance(path, dict):
                    node, path = self.outer_path(path)
                    for child, child_node in node.inner.items():
                        state[child] = child_node.topology_state(path)
                else:
                    node = self.get_path(path)
                    for child, child_node in node.inner.items():
                        state[child] = child_node.get_value()
            elif isinstance(path, dict):
                node, path = self.outer_path(path)
                state[key] = node.topology_state(path)
            else:
                state[key] = self.get_path(path).get_value()
        return state

    def schema_topology(self, schema, topology):
        """
        Fill in the structure of the given schema with the connected stores
        according to the given topology.
        """

        state = {}

        if self.leaf or schema == '**':
            state = self
        elif not schema.get('_output'):
            for key, subschema in schema.items():
                path = topology.get(key)
                if key == '*':
                    if isinstance(path, dict):
                        node, path = self.outer_path(path)
                        for child, child_node in node.inner.items():
                            state[child] = child_node.schema_topology(
                                subschema, path)
                    else:
                        node = self.get_path(path)
                        for child, child_node in node.inner.items():
                            state[child] = child_node.schema_topology(
                                subschema, {})
                elif key == '_divider':
                    pass
                elif isinstance(path, dict):
                    node, path = self.outer_path(path)
                    state[key] = node.schema_topology(subschema, path)
                else:
                    if path is None:
                        path = (key,)
                    node = self.get_path(path)
                    if node:
                        state[key] = node.schema_topology(subschema, {})
                    else:
                        # node is None, it was likely deleted
                        print('{} is None'.format(path))

        return state

    def state_for(self, path, keys):
        """
        Get the value of a state at a given path
        """

        state = self.get_path(path)
        if state is None:
            return {}
        if keys and keys[0] == '*':
            return state.get_value()
        return {
            key: state.inner_value(key)
            for key in keys}

    def depth(self, path=(), filter_function=None):
        """
        Create a mapping of every path in the tree to the node living at
        that path in the tree. An optional `filter` argument is a function
        that can declares the instances that will be returned, for example:
        * filter=lambda x: isinstance(x.value, Process)
        """
        base = []
        if filter_function is None or filter_function(self):
            base += [(path, self)]

        for key, child in self.inner.items():
            down = tuple(path + (key,))
            base += child.depth(down, filter_function)
        return base

    def add_node(self, path, node):
        """ Add a node instance at the provided path """
        target = self._establish_path(path[:-1], {})
        if target.get_value() and path[-1] in target.get_value():
            # this path already exists, update it
            self.apply_update({path[-1]: node.get_value()})
        else:
            node.outer = target
            target.inner.update({path[-1]: node})
        return target

    def outer_path(self, path, source=None):
        """
        Address a topology with the `_path` keyword if present,
        establishing a path to this node and using it as the
        starting point for future path operations.
        """

        node = self
        if '_path' in path:
            node = self._establish_path(
                path['_path'],
                {},
                source=source)
            path = without(path, '_path')

        return node, path

    def _apply_subschema_path(self, path):
        """Apply ``self.subschema`` at ``path``."""
        if path:
            inner = self.inner[path[0]]
            if self.subschema:
                subtopology = self.subtopology or {}
                inner._topology_ports(
                    self.subschema,
                    subtopology,
                    source=self.path_for() + ('*',))
            inner._apply_subschema_path(path[1:])

    def _apply_subschema(self, subschema=None, subtopology=None):
        """
        Apply a subschema to all inner nodes (either provided or from this
        node's personal subschema) as governed by the given/personal
        subtopology.
        """

        if subschema is None:
            subschema = self.subschema
        if subtopology is None:
            subtopology = self.subtopology or {}

        inner = list(self.inner.values())

        for child in inner:
            child._topology_ports(
                subschema,
                subtopology,
                source=self.path_for() + ('*',))

    def _apply_subschemas(self):
        """
        Apply all subschemas from all nodes at this point or lower in the tree.
        """

        if self.subschema:
            self._apply_subschema()
        for child in self.inner.values():
            child._apply_subschemas()

    def _update_subschema(self, path, subschema):
        """
        Merge a new subschema into an existing subschema at the given path.
        """

        target = self.get_path(path)
        if target.subschema is None:
            target.subschema = subschema
        else:
            target.subschema = deep_merge(
                target.subschema,
                subschema)
        return target

    def _establish_path(self, path, config, source=None):
        """
        Create a node at the given path if it does not exist, then
        apply a config to it.

        Paths can include '..' to go up a level (which raises an exception
        if that level does not exist).
        """

        if len(path) > 0:
            path_step = path[0]
            remaining = path[1:]

            if path_step == '..':
                if not self.outer:
                    raise Exception(
                        'outer does not exist for path: {}'.format(path))

                return self.outer._establish_path(
                    remaining,
                    config,
                    source=source)
            elif isinstance(self.value, Process):
                towards = topology_path(self.topology, path)
                if towards:
                    target = self.outer._establish_path(towards[0], config, source=source)
                    return target._establish_path(towards[1], config, source=source)
            else:
                if path_step not in self.inner:
                    self.inner[path_step] = Store(
                        {}, outer=self, source=source)

                return self.inner[path_step]._establish_path(
                    remaining,
                    config,
                    source=source)
        else:
            self._apply_config(config, source=source)
            return self

    def _topology_ports(self, schema, topology, source=None):
        """
        Distribute a schema into the tree by mapping its ports
        according to the given topology.
        """

        source = source or self.path_for()

        if set(schema.keys()) & self.schema_keys:
            self.get_path(topology)._apply_config(schema)
        else:
            mismatch_topology = (
                set(topology.keys()) - set(schema.keys()))
            mismatch_schema = (
                set(schema.keys()) - set(topology.keys()))
            if mismatch_topology:
                raise Exception(
                    'topology for the process {} \n at path {} uses '
                    'undeclared ports: {}'.format(
                        source, self.path_for(), str(mismatch_topology)))
            if mismatch_schema:
                log.info(
                    'process %s has ports that are not included in '
                    'the topology: %s', str(source), str(mismatch_schema))

            for port, subschema in schema.items():
                path = topology.get(port, (port,))

                if port == '*':
                    subschema_config = {
                        '_subschema': subschema}
                    if isinstance(path, dict):
                        node, subpath = self.outer_path(
                            path, source=source)
                        node._merge_subtopology(subpath)
                        node._apply_config(subschema_config)
                    else:
                        node = self._establish_path(
                            path,
                            subschema_config,
                            source=source)
                    node._apply_subschema()
                    node.apply_defaults()

                elif isinstance(path, dict):
                    node, subpath = self.outer_path(
                        path, source=source)

                    node._topology_ports(
                        subschema,
                        subpath,
                        source=source)

                else:
                    self._establish_path(
                        path,
                        subschema,
                        source=source)

    def _generate_paths(
            self, processes: Processes, flow: Flow, topology: Topology):
        """Set up state :term:`hierarchy` with stores.

        Recursively creates the entire state hierarchy rooted at
        ``self``.

        .. WARNING::

           This method is public since it is used by
           :py:func:`generate_state`, but it should not be used from
           outside this module. It is not part of the supported API.

        Args:
            processes: Map from process names to process objects,
                embedded within nested dictionaries representing the
                hierarchy nodes rooted at ``self``. Note that steps
                should be included here too.
            flow: Map from step names to relative paths to step
                dependencies, embedded within nested dictionaries
                representing the hierarchy nodes rooted at ``self``.
            topology: The topology.
        """
        for key, subprocess in processes.items():
            subtopology = topology[key]
            subflow = flow.get(key) if flow else None
            if isinstance(subprocess, Process):
                schema = {
                    '_value': subprocess,
                    '_updater': 'set',
                    '_topology': subtopology,
                    '_serializer': 'process'
                }
                if subflow is not None:
                    schema['_flow'] = subflow
                process_state = Store(schema, outer=self)

                self.inner[key] = process_state

                subprocess.schema = subprocess.get_schema()

                self._topology_ports(
                    subprocess.schema,
                    subtopology,
                    source=self.path_for() + (key,))
            else:
                if key not in self.inner:
                    self.inner[key] = Store({}, outer=self)
                self.inner[key]._generate_paths(
                    subprocess,
                    subflow,
                    subtopology,
                )

    def build_topology_views(self):
        if self.leaf:
            if isinstance(self.value, Process):
                self.topology_view = self.outer.schema_topology(
                    self.value.schema,
                    self.topology)
        else:
            for inner in self.inner.values():
                inner.build_topology_views()

    def generate(self, path, processes, steps, flow, topology, initial_state):
        """
        Generate a subtree of this store at the given path.
        The processes will be mapped into locations in the tree by the
        topology, and once everything is constructed the initial_state
        will be applied.
        """

        target = self._establish_path(path, {})
        target._generate_paths(processes, flow, topology)
        target._generate_paths(steps, flow, topology)
        target._apply_subschemas()
        target.set_value(initial_state)
        target.apply_defaults()
