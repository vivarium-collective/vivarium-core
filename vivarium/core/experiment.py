'''
==========================================
Experiment and Store Classes
==========================================
'''

import os
import copy
import math
import datetime
import time as clock
import uuid

import numpy as np
import logging as log

import pprint

from multiprocessing import Pool
from typing import Any, Dict

from vivarium.library.units import Quantity
from vivarium.library.dict_utils import (
    deep_merge,
    deep_merge_multi_update,
    MULTI_UPDATE_KEY,
)
from vivarium.core.emitter import get_emitter
from vivarium.core.process import (
    Generator,
    Process,
    ParallelProcess,
    serialize_dictionary,
)
from vivarium.library.topology import (
    get_in, delete_in, assoc_path,
    without, update_in, inverse_topology
)
from vivarium.core.registry import (
    divider_registry,
    updater_registry,
    serializer_registry,
)

pretty = pprint.PrettyPrinter(indent=2)

def pp(x):
    pretty.pprint(x)

def pf(x):
    return pretty.pformat(x)


EMPTY_UPDATES = None, None, None

log.basicConfig(level=os.environ.get("LOGLEVEL", log.WARNING))


def starts_with(l, sub):
    return len(sub) <= len(l) and all(
        l[i] == el
        for i, el in enumerate(sub))

# Store
def key_for_value(d, looking):
    found = None
    for key, value in d.items():
        if looking == value:
            found = key
            break
    return found


def depth(tree, path=()):
    '''
    Create a mapping of every path in the tree to the node living at
    that path in the tree.
    '''

    base = {}

    for key, inner in tree.items():
        down = tuple(path + (key,))
        if isinstance(inner, dict):
            base.update(depth(inner, down))
        else:
            base[down] = inner

    return base


def schema_for(port, keys, initial_state, default=0.0, updater='accumulate'):
    return {
        key: {
            '_default': initial_state.get(
                port, {}).get(key, default),
            '_updater': updater}
        for key in keys}


def always_true(_):
    return True


def identity(y):
    return y


class Store(object):
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
        self.updater_definition = None
        self.value = None
        self.units = None
        self.divider = None
        self.emit = False
        self.sources = {}
        self.leaf = False
        self.serializer = None

        self.apply_config(config, source)

    def check_default(self, new_default):
        defaults_equal = False
        if self.default is not None:
            self_default_comp = self.default
            new_default_comp = new_default
            if isinstance(self_default_comp, np.ndarray):
                self_default_comp = self.default.tolist()
            if isinstance(new_default_comp, np.ndarray):
                new_default_comp = self.default.tolist()
            defaults_equal = self_default_comp == new_default_comp
        if defaults_equal:
            if (
                not isinstance(new_default, np.ndarray)
                and not isinstance(self.default, np.ndarray)
                and new_default == 0
                and self.default != 0
            ):
                log.debug('_default schema conflict: {} and {}. selecting {}'.format(
                    self.default, new_default, self.default))
                return self.default
            log.debug('_default schema conflict: {} and {}. selecting {}'.format(
                self.default, new_default, new_default))
        return new_default

    def check_value(self, new_value):
        if self.value is not None and new_value != self.value:
            raise Exception('_value schema conflict: {} and {}'.format(new_value, self.value))
        return new_value

    def merge_subtopology(self, subtopology):
        self.subtopology = deep_merge(self.subtopology, subtopology)

    def apply_subschema_config(self, subschema):
        self.subschema = deep_merge(
            self.subschema,
            subschema)

    def apply_config(self, config, source=None):
        '''
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
        * _emit - whether or not to emit the values under this point in the tree.
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
        '''


        if '*' in config:
            self.apply_subschema_config(config['*'])
            config = without(config, '*')

        if '_subschema' in config:
            if source:
                self.sources[source] = config['_subschema']
            self.apply_subschema_config(config['_subschema'])
            config = without(config, '_subschema')

        if '_subtopology' in config:
            self.merge_subtopology(config['_subtopology'])
            config = without(config, '_subtopology')

        if '_divider' in config:
            self.divider = config['_divider']
            if isinstance(self.divider, str):
                self.divider = divider_registry.access(self.divider)
            if isinstance(self.divider, dict) and isinstance(self.divider['divider'], str):
                self.divider['divider'] = divider_registry.access(self.divider['divider'])
            config = without(config, '_divider')

        if self.schema_keys & set(config.keys()):
            if self.inner:
                raise Exception('trying to assign leaf values to a branch at: {}'.format(self.path_for()))
            self.leaf = True
            # self.units = config.get('_units', self.units)
            if '_serializer' in config:
                self.serializer = config['_serializer']
                if isinstance(self.serializer, str):
                    self.serializer = serializer_registry.access(self.serializer)

            if '_default' in config:
                self.default = self.check_default(config.get('_default'))
                if isinstance(self.default, Quantity):
                    self.units = self.default.units
                if isinstance(self.default, np.ndarray):
                    self.serializer = self.serializer or serializer_registry.access('numpy')

            if '_value' in config:
                self.value = self.check_value(config.get('_value'))
                if isinstance(self.value, Quantity):
                    self.units = self.value.units

            self.updater_definition = config.get(
                '_updater',
                self.updater_definition or 'accumulate',
            )

            self.properties = deep_merge(
                self.properties,
                config.get('_properties', {}))

            self.emit = config.get('_emit', self.emit)

            if source:
                self.sources[source] = config

        else:
            if self.leaf and config:
                raise Exception('trying to assign create inner for leaf node: {}'.format(self.path_for()))

            # self.value = None

            for key, child in config.items():
                if key not in self.inner:
                    self.inner[key] = Store(child, outer=self, source=source)
                else:
                    self.inner[key].apply_config(child, source=source)

    def get_updater(self, update):
        updater_definition = self.updater_definition
        if isinstance(update, dict) and '_updater' in update:
            updater_definition = update['_updater']
        port_mapping = None
        if isinstance(updater_definition, dict):
            updater = updater_definition['updater']
            port_mapping = updater_definition['port_mapping']
        else:
            updater = updater_definition

        if isinstance(updater, str):
            updater = updater_registry.access(updater)
        return updater, port_mapping

    def get_config(self, sources=False):
        '''
        Assemble a dictionary representation of the config for this node.
        A desired property is that the node can be exactly recreated by
        applying the resulting config to an empty node again.
        '''

        config = {}

        if self.properties:
            config['_properties'] = self.properties
        if self.subschema:
            config['_subschema'] = self.subschema
        if self.subtopology:
            config['_subtopology'] = self.subtopology
        if self.divider:
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
            if self.updater_definition:
                config['_updater'] = self.updater_definition
            if self.units:
                config['_units'] = self.units
            if self.emit:
                config['_emit'] = self.emit

        return config

    def top(self):
        '''
        Find the top of this tree.
        '''

        if self.outer:
            return self.outer.top()
        else:
            return self

    def path_for(self):
        '''
        Find the path to this node.
        '''

        if self.outer:
            key = key_for_value(self.outer.inner, self)
            above = self.outer.path_for()
            return above + (key,)
        else:
            return tuple()

    def get_value(self, condition=None, f=None):
        '''
        Pull the values out of the tree in a structure symmetrical to the tree.
        '''

        if self.inner:
            if condition is None:
                condition = always_true

            if f is None:
                f = identity

            return {
                key: f(child.get_value(condition, f))
                for key, child in self.inner.items()
                if condition(child)}
        else:
            if self.subschema:
                return {}
            else:
                return self.value

    def get_path(self, path):
        '''
        Get the node at the given path relative to this node.
        '''

        if path:
            step = path[0]
            if step == '..':
                child = self.outer
            else:
                child = self.inner.get(step)

            if child:
                return child.get_path(path[1:])
            else:
                # TODO: more handling for bad paths?
                # TODO: check deleted?
                return None
        else:
            return self

    def get_paths(self, paths):
        return {
            key: self.get_path(path)
            for key, path in paths.items()}

    def get_values(self, paths):
        return {
            key: self.get_in(path)
            for key, path in paths.items()}

    def get_in(self, path):
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
        data = {}
        if self.inner:
            for key, child in self.inner.items():
                child_data = child.emit_data()
                if child_data is not None or child_data == 0:
                    data[key] = child_data
            return data
        else:
            if self.emit:
                if self.serializer:
                    return self.serializer.serialize(self.value)
                elif isinstance(self.value, Process):
                    return self.value.pull_data()
                else:
                    if self.units:
                        return self.value.to(self.units).magnitude
                    else:
                        return self.value

    def delete_path(self, path):
        '''
        Delete the subtree at the given path.
        '''

        if not path:
            self.inner = {}
            self.value = None
            return self
        else:
            target = self.get_path(path[:-1])
            remove = path[-1]
            if remove in target.inner:
                lost = target.inner[remove]
                del target.inner[remove]
                return lost

    def divide_value(self):
        '''
        Apply the divider for each node to the value in that node to
        assemble two parallel divided states of this subtree.
        '''

        if self.divider:
            # divider is either a function or a dict with topology
            if isinstance(self.divider, dict):
                divider = self.divider['divider']
                topology = self.divider['topology']
                state = self.outer.get_values(topology)
                return divider(self.get_value(), state)
            else:
                return self.divider(self.get_value())
        elif self.inner:
            daughters = [{}, {}]
            for key, child in self.inner.items():
                division = child.divide_value()
                if division:
                    for daughter, divide in zip(daughters, division):
                        daughter[key] = divide
            return daughters

    def reduce(self, reducer, initial=None):
        '''
        Call the reducer on each node accumulating over the result.
        '''

        value = initial

        for path, node in self.depth():
            value = reducer(value, path, node)
        return value

    def set_value(self, value):
        '''
        Set the value for the given tree elements directly instead of using
        the updaters from their nodes.
        '''

        if self.inner or self.subschema:
            for child, inner_value in value.items():
                if child not in self.inner:
                    if self.subschema:
                        self.inner[child] = Store(self.subschema, self)
                    else:
                        pass

                        # TODO: continue to ignore extra keys?
                        # print("setting value that doesn't exist in tree {} {}".format(
                        #     child, inner_value))

                if child in self.inner:
                    self.inner[child].set_value(inner_value)
        else:
            self.value = value

    def apply_defaults(self):
        '''
        If value is None, set to default.
        '''
        if self.inner:
            for child in self.inner.values():
                child.apply_defaults()
        else:
            if self.value is None:
                self.value = self.default

    def apply_update(self, update, process_topology=None, state=None):
        '''
        Given an arbitrary update, map all the values in that update
        to their positions in the tree where they apply, and update
        these values using each node's `_updater`.

        There are five special update keys:

        * `_updater` - Override the default updater with any updater you want.
        * `_delete` - The value here is a list of paths (tuples) to delete from
          the tree.
        * `_add` - Adds states into the subtree, given a list of dicts containing:

            * path - Path to the added state key.
            * state - The value of the added state.

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
        '''

        if isinstance(update, dict) and MULTI_UPDATE_KEY in update:
            # apply multiple updates to same node
            multi_update = update[MULTI_UPDATE_KEY]
            assert isinstance(multi_update, list)
            for update_value in multi_update:
                self.apply_update(update_value, process_topology, state)
            return EMPTY_UPDATES

        elif self.inner or self.subschema:
            # Branch update: this node has an inner

            process_updates, topology_updates, deletions = {}, {}, []
            update = dict(update)  # avoid mutating the caller's dict

            delete_paths = update.pop('_delete', None)
            if delete_paths is not None:
                # delete a list of paths
                here = self.path_for()
                for path in delete_paths:
                    self.delete_path(path)
                    deletions.append(tuple(here + path))

            add_entries = update.pop('_add', None)
            if add_entries is not None:
                # add a list of sub-states
                for added in add_entries:
                    path = added['path']
                    state = added['state']

                    target = self.establish_path(path, {})
                    self.apply_subschema_path(path)
                    target.apply_defaults()
                    target.set_value(state)

            generate_entries = update.pop('_generate', None)
            if generate_entries is not None:
                # generate a list of new processes
                for generate in generate_entries:
                    path = generate.get('path', tuple())

                    self.generate(
                        path,
                        generate['processes'],
                        generate['topology'],
                        generate['initial_state'])

                    assoc_path(
                        process_updates,
                        path,
                        generate['processes'])

                    assoc_path(
                        topology_updates,
                        path,
                        generate['topology'])

                    self.apply_subschema_path(path)
                    self.get_path(path).apply_defaults()

            divide = update.pop('_divide', None)
            if divide is not None:
                # use dividers to find initial states for daughters
                mother = divide['mother']
                daughters = divide['daughters']
                initial_state = self.inner[mother].get_value(
                    condition=lambda child: not (isinstance(child.value, Process)),
                    f=lambda child: copy.deepcopy(child))
                states = self.inner[mother].divide_value()

                for daughter, state in zip(daughters, states):
                    # daughter_id = daughter['daughter']

                    # use initial state as default, merge in divided values
                    initial_state = deep_merge(
                        initial_state,
                        state)

                    path = daughter['path']

                    self.generate(
                        path,
                        daughter['processes'],
                        daughter['topology'],
                        daughter['initial_state'])

                    assoc_path(
                        process_updates,
                        path,
                        daughter['processes'])

                    assoc_path(
                        topology_updates,
                        path,
                        daughter['topology'])

                    self.apply_subschema_path(path)
                    target = self.get_path(path)
                    target.apply_defaults()
                    target.set_value(initial_state)

                here = self.path_for()
                mother_path = (mother,)
                self.delete_path(mother_path)
                deletions.append(tuple(here + mother_path))

            for key, value in update.items():
                if key in self.inner:
                    inner = self.inner[key]
                    inner_topology, inner_processes, inner_deletions = inner.apply_update(
                        value, process_topology, state)

                    if inner_topology:
                        topology_updates = deep_merge(
                            topology_updates,
                            {key: inner_topology})

                    if inner_processes:
                        process_updates = deep_merge(
                            process_updates,
                            {key: inner_processes})

                    if inner_deletions:
                        deletions += inner_deletions

            return topology_updates, process_updates, deletions

        else:
            # Leaf update: this node has no inner

            updater, port_mapping = self.get_updater(update)
            state_dict = None

            if isinstance(update, dict) and '_reduce' in update:
                assert port_mapping is None
                reduction = update['_reduce']
                top = self.get_path(reduction.get('from'))
                update = top.reduce(
                    reduction['reducer'],
                    initial=reduction['initial'])

            if isinstance(update, dict) and self.schema_keys & set(update.keys()):
                if '_updater' in update:
                    update = update.get('_value', self.default)

            if port_mapping is not None:
                updater_topology = {
                    updater_port: process_topology[proc_port]
                    for updater_port, proc_port in port_mapping.items()}

                state_dict = state.outer.topology_state(
                    updater_topology)

            self.value = updater(self.value, update, state_dict)

            return EMPTY_UPDATES

    def inner_value(self, key):
        '''
        Get the value of an inner state
        '''

        if key in self.inner:
            return self.inner[key].get_value()

    def topology_state(self, topology):
        '''
        Fill in the structure of the given topology with the values at all
        the paths the topology points at. Essentially, anywhere in the topology
        that has a tuple path will be filled in with the value at that path.

        This is the inverse function of the standalone `inverse_topology`.
        '''

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
        '''
        Fill in the structure of the given schema with the values located according
        to the given topology.
        '''

        state = {}

        if self.leaf:
            state = self.get_value()
        else:
            for key, subschema in schema.items():
                path = topology.get(key)
                if key == '*':
                    if isinstance(path, dict):
                        node, path = self.outer_path(path)
                        for child, child_node in node.inner.items():
                            state[child] = child_node.schema_topology(subschema, path)
                    else:
                        node = self.get_path(path)
                        for child, child_node in node.inner.items():
                            state[child] = child_node.schema_topology(subschema, {})
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
        '''
        Get the value of a state at a given path
        '''

        state = self.get_path(path)
        if state is None:
            return {}
        elif keys and keys[0] == '*':
            return state.get_value()
        else:
            return {
                key: state.inner_value(key)
                for key in keys}

    def depth(self, path=()):
        '''
        Create a mapping of every path in the tree to the node living at
        that path in the tree.
        '''

        base = [(path, self)]
        for key, child in self.inner.items():
            down = tuple(path + (key,))
            base += child.depth(down)
        return base

    def processes(self, path=()):
        return {
            path: state
            for path, state in self.depth()
            if state.value and isinstance(state.value, Process)}

    def apply_subschema_path(self, path):
        if path:
            inner = self.inner[path[0]]
            if self.subschema:
                subtopology = self.subtopology or {}
                inner.topology_ports(
                    self.subschema,
                    subtopology,
                    source=self.path_for() + ('*',))
            inner.apply_subschema_path(path[1:])

    def apply_subschema(self, subschema=None, subtopology=None, source=None):
        '''
        Apply a subschema to all inner nodes (either provided or from this
        node's personal subschema) as governed by the given/personal
        subtopology.
        '''

        if subschema is None:
            subschema = self.subschema
        if subtopology is None:
            subtopology = self.subtopology or {}

        inner = list(self.inner.items())

        for child_key, child in inner:
            child.topology_ports(
                subschema,
                subtopology,
                source=self.path_for() + ('*',))

    def apply_subschemas(self):
        '''
        Apply all subschemas from all nodes at this point or lower in the tree.
        '''

        if self.subschema:
            self.apply_subschema()
        for child in self.inner.values():
            child.apply_subschemas()

    def update_subschema(self, path, subschema):
        '''
        Merge a new subschema into an existing subschema at the given path.
        '''

        target = self.get_path(path)
        if target.subschema is None:
            target.subschema = subschema
        else:
            target.subschema = deep_merge(
                target.subschema,
                subschema)
        return target

    def establish_path(self, path, config, source=None):
        '''
        Create a node at the given path if it does not exist, then
        apply a config to it.

        Paths can include '..' to go up a level (which raises an exception
        if that level does not exist).
        '''

        if len(path) > 0:
            path_step = path[0]
            remaining = path[1:]

            if path_step == '..':
                if not self.outer:
                    raise Exception('outer does not exist for path: {}'.format(path))

                return self.outer.establish_path(
                    remaining,
                    config,
                    source=source)
            else:
                if path_step not in self.inner:
                    self.inner[path_step] = Store({}, outer=self, source=source)

                return self.inner[path_step].establish_path(
                    remaining,
                    config,
                    source=source)
        else:
            self.apply_config(config, source=source)
            return self

    def outer_path(self, path, source=None):
        '''
        Address a topology with the `_path` keyword if present,
        establishing a path to this node and using it as the
        starting point for future path operations.
        '''

        node = self
        if '_path' in path:
            node = self.establish_path(
                path['_path'],
                {},
                source=source)
            path = without(path, '_path')

        return node, path

    def topology_ports(self, schema, topology, source=None):
        '''
        Distribute a schema into the tree by mapping its ports
        according to the given topology.
        '''

        source = source or self.path_for()

        if set(schema.keys()) & self.schema_keys:
            self.get_path(topology).apply_config(schema)
        else:
            mismatch_topology = (
                set(topology.keys()) - set(schema.keys()))
            mismatch_schema = (
                set(schema.keys()) - set(topology.keys()))
            if mismatch_topology:
                raise Exception(
                    'the topology for process {} at path {} uses undeclared ports: {}'.format(
                        source, self.path_for(), mismatch_topology))
            if mismatch_schema:
                log.info(
                    'process {} has ports that are not included in the topology: {}'.format(
                        source, mismatch_schema))

            for port, subschema in schema.items():
                path = topology.get(port, (port,))

                if port == '*':
                    subschema_config = {
                        '_subschema': subschema}
                    if isinstance(path, dict):
                        node, path = self.outer_path(
                            path, source=source)
                        node.merge_subtopology(path)
                        node.apply_config(subschema_config)
                    else:
                        node = self.establish_path(
                            path,
                            subschema_config,
                            source=source)
                    node.apply_subschema()
                    node.apply_defaults()

                elif isinstance(path, dict):
                    node, path = self.outer_path(
                        path, source=source)

                    node.topology_ports(
                        subschema,
                        path,
                        source=source)

                else:
                    self.establish_path(
                        path,
                        subschema,
                        source=source)

    def generate_paths(self, processes, topology):
        for key, subprocess in processes.items():
            subtopology = topology[key]
            if isinstance(subprocess, Process):
                process_state = Store({
                    '_value': subprocess,
                    '_updater': 'set',
                    '_serializer': 'process'}, outer=self)
                self.inner[key] = process_state

                subprocess.schema = subprocess.get_schema()
                self.topology_ports(
                    subprocess.schema,
                    subtopology,
                    source=self.path_for() + (key,))
            else:
                if key not in self.inner:
                    self.inner[key] = Store({}, outer=self)
                self.inner[key].generate_paths(
                    subprocess,
                    subtopology)

    def generate(self, path, processes, topology, initial_state):
        '''
        Generate a subtree of this store at the given path.
        The processes will be mapped into locations in the tree by the
        topology, and once everything is constructed the initial_state
        will be applied.
        '''

        target = self.establish_path(path, {})
        target.generate_paths(processes, topology)
        target.set_value(initial_state)
        target.apply_subschemas()
        target.apply_defaults()


def invert_topology(update, args):
    path, topology = args
    return inverse_topology(path[:-1], update, topology)


def generate_state(processes, topology, initial_state):
    state = Store({})
    state.generate_paths(processes, topology)
    state.apply_subschemas()
    state.set_value(initial_state)
    state.apply_defaults()

    return state


def timestamp(dt=None):
    if not dt:
        dt = datetime.datetime.now()
    return "%04d%02d%02d.%02d%02d%02d" % (
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second)


def invoke_process(process, interval, states):
    return process.next_update(interval, states)


class Defer(object):
    def __init__(self, defer, f, args):
        self.defer = defer
        self.f = f
        self.args = args

    def get(self):
        return self.f(
            self.defer.get(),
            self.args)


class InvokeProcess(object):
    def __init__(self, process, interval, states):
        self.process = process
        self.interval = interval
        self.states = states
        self.update = invoke_process(
            self.process,
            self.interval,
            self.states)

    def get(self, timeout=0):
        return self.update


class MultiInvoke(object):
    def __init__(self, pool):
        self.pool = pool

    def invoke(self, process, interval, states):
        args = (process, interval, states)
        result = self.pool.apply_async(invoke_process, args)
        return result


class Experiment(object):
    def __init__(self, config):
        # type: (Dict[str, Any]) -> None
        """Defines simulations

        Arguments:
            config (dict): A dictionary of configuration options. The
                required options are:

                * **processes** (:py:class:`dict`): A dictionary that
                    maps :term:`process` names to process objects. You
                    will usually get this from the ``processes``
                    attribute of the dictionary from
                    :py:meth:`vivarium.core.process.Generator.generate`.
                * **topology** (:py:class:`dict`): A dictionary that
                    maps process names to sub-dictionaries. These
                    sub-dictionaries map the process's port names to
                    tuples that specify a path through the :term:`tree`
                    from the :term:`compartment` root to the
                    :term:`store` that will be passed to the process for
                    that port.

                The following options are optional:

                * **experiment_id** (:py:class:`uuid.UUID` or
                    :py:class:`str`): A unique identifier for the
                    experiment. A UUID will be generated if none is
                    provided.
                * **description** (:py:class:`str`): A description of
                    the experiment. A blank string by default.
                * **initial_state** (:py:class:`dict`): By default an
                    empty dictionary, this is the initial state of the
                    simulation.
                * **emitter** (:py:class:`dict`): An emitter
                    configuration which must conform to the
                    specification in the documentation for
                    :py:func:`vivarium.core.emitter.get_emitter`. The
                    experiment ID will be added to the dictionary you
                    provide as the value for the key ``experiment_id``.
        """
        self.config = config
        self.experiment_id = config.get(
            'experiment_id', str(uuid.uuid1()))
        self.processes = config['processes']
        self.topology = config['topology']
        self.initial_state = config.get('initial_state', {})
        self.emit_step = config.get('emit_step')

        # display settings
        self.experiment_name = config.get('experiment_name', self.experiment_id)
        self.description = config.get('description', '')
        self.display_info = config.get('display_info', True)
        self.progress_bar = config.get('progress_bar', True)
        self.time_created = timestamp()
        if self.display_info:
            self.print_display()

        # parallel settings
        self.invoke = config.get('invoke', InvokeProcess)
        self.parallel = {}

        # get a mapping of all paths to processes
        self.process_paths = {}
        self.deriver_paths = {}
        self.find_process_paths(self.processes)

        # initialize the state
        self.state = generate_state(
            self.processes,
            self.topology,
            self.initial_state)

        # emitter settings
        emitter_config = config.get('emitter', 'timeseries')
        if isinstance(emitter_config, str):
            emitter_config = {'type': emitter_config}
        emitter_config['experiment_id'] = self.experiment_id
        self.emitter = get_emitter(emitter_config)

        self.local_time = 0.0

        # run the derivers
        self.send_updates([])

        # run the emitter
        self.emit_configuration()
        self.emit_data()

        # logging information
        log.info('experiment {}'.format(self.experiment_id))

        log.info('\nPROCESSES:')
        log.info(pf(self.processes))

        log.info('\nTOPOLOGY:')
        log.info(pf(self.topology))

        # log.info('\nSTATE:')
        # log.info(pf(self.state.get_value()))
        #
        # log.info('\nCONFIG:')
        # log.info(pf(self.state.get_config(True)))

    def find_process_paths(self, processes):
        tree = depth(processes)
        for path, process in tree.items():
            if process.is_deriver():
                self.deriver_paths[path] = process
            else:
                self.process_paths[path] = process

    def emit_configuration(self):
        data = {
            'time_created': self.time_created,
            'experiment_id': self.experiment_id,
            'name': self.experiment_name,
            'description': self.description,
            'topology': self.topology,
            # TODO -- handle large parameter sets in self.processes to meet mongoDB limit
            # 'processes': serialize_dictionary(self.processes),
            # 'state': self.state.get_config()
        }
        emit_config = {
            'table': 'configuration',
            'data': data}
        self.emitter.emit(emit_config)

    def invoke_process(self, process, path, interval, states):
        if process.parallel:
            # add parallel process if it doesn't exist
            if path not in self.parallel:
                self.parallel[path] = ParallelProcess(process)
            # trigger the computation of the parallel process
            self.parallel[path].update(interval, states)

            return self.parallel[path]
        else:
            # if not parallel, perform a normal invocation
            return self.invoke(process, interval, states)

    def process_update(self, path, process, interval):
        state = self.state.get_path(path)
        process_topology = get_in(self.topology, path)

        # translate the values from the tree structure into the form
        # that this process expects, based on its declared topology
        states = state.outer.schema_topology(process.schema, process_topology)

        update = self.invoke_process(
            process,
            path,
            interval,
            states)

        absolute = Defer(update, invert_topology, (path, process_topology))

        return absolute, process_topology, state

    def apply_update(self, update, process_topology, state):
        topology_updates, process_updates, deletions = self.state.apply_update(
            update, process_topology, state)

        if topology_updates:
            self.topology = deep_merge(self.topology, topology_updates)

        if process_updates:
            self.processes = deep_merge(self.processes, process_updates)
            self.find_process_paths(process_updates)

        if deletions:
            for deletion in deletions:
                self.delete_path(deletion)

    def delete_path(self, deletion):
        delete_in(self.processes, deletion)
        delete_in(self.topology, deletion)

        for path in list(self.process_paths.keys()):
            if starts_with(path, deletion):
                del self.process_paths[path]

        for path in list(self.deriver_paths.keys()):
            if starts_with(path, deletion):
                del self.deriver_paths[path]

    def run_derivers(self):
        paths = list(self.deriver_paths.keys())
        for path in paths:
            # deriver could have been deleted by another deriver
            deriver = self.deriver_paths.get(path)
            if deriver:
                # timestep shouldn't influence derivers
                update, process_topology, state = self.process_update(
                    path, deriver, 0)
                self.apply_update(update.get(), process_topology, state)

    def emit_data(self):
        data = self.state.emit_data()
        data.update({
            'time': self.local_time})
        emit_config = {
            'table': 'history',
            'data': serialize_dictionary(data)}
        self.emitter.emit(emit_config)

    def send_updates(self, update_tuples):
        for update_tuple in update_tuples:
            update, process_topology, state = update_tuple
            self.apply_update(update.get(), process_topology, state)

        self.run_derivers()

    def update(self, interval):
        """ Run each process for the given interval and update the related states. """

        time = 0
        emit_time = self.emit_step
        clock_start = clock.time()

        def empty_front(t):
            return {
                'time': t,
                'update': {}}

        # keep track of which processes have simulated until when
        front = {}

        while time < interval:
            full_step = math.inf

            # find any parallel processes that were removed and terminate them
            for terminated in self.parallel.keys() - self.process_paths.keys():
                self.parallel[terminated].end()
                del self.parallel[terminated]

            # setup a way to track how far each process has simulated in time
            front = {
                path: progress
                for path, progress in front.items()
                if path in self.process_paths}

            # go through each process and find those that are able to update
            # based on their current time being less than the global time.
            for path, process in self.process_paths.items():
                if path not in front:
                    front[path] = empty_front(time)
                process_time = front[path]['time']

                if process_time <= time:
                    future = min(process_time + process.local_timestep(), interval)
                    timestep = future - process_time

                    # calculate the update for this process
                    update = self.process_update(path, process, timestep)

                    # store the update to apply at its projected time
                    if timestep < full_step:
                        full_step = timestep
                    front[path]['time'] = future
                    front[path]['update'] = update

            if full_step == math.inf:
                # no processes ran, jump to next process
                next_event = interval
                for path in front.keys():
                    if front[path]['time'] < next_event:
                        next_event = front[path]['time']
                time = next_event
            else:
                # at least one process ran, apply updates and continue
                future = time + full_step

                updates = []
                paths = []
                for path, advance in front.items():
                    if advance['time'] <= future:
                        new_update = advance['update']
                        # new_update['_path'] = path
                        updates.append(new_update)
                        advance['update'] = {}
                        paths.append(path)

                self.send_updates(updates)

                time = future
                self.local_time += full_step

                if self.progress_bar:
                    print_progress_bar(time, interval)
                if self.emit_step is None:
                    self.emit_data()
                elif emit_time <= time:
                    while emit_time <= time:
                        self.emit_data()
                        emit_time += self.emit_step

        # post-simulation
        for process_name, advance in front.items():
            assert advance['time'] == time == interval
            assert len(advance['update']) == 0

        clock_finish = clock.time() - clock_start

        if self.display_info:
            self.print_summary(clock_finish)

    def end(self):
        for parallel in self.parallel.values():
            parallel.end()

    def print_display(self):
        date, time = self.time_created.split('.')
        print('\nExperiment ID: {}'.format(self.experiment_id))
        print('Created: {} at {}'.format(
            date[4:6] + '/' + date[6:8] + '/' + date[0:4],
            time[0:2] + ':' + time[2:4] + ':' + time[4:6]))
        if self.experiment_name is not self.experiment_id:
            print('Name: {}'.format(self.experiment_name))
        if self.description:
            print('Description: {}'.format(self.description))

    def print_summary(self, clock_finish):
        if clock_finish < 1:
            print('Completed in {:.6f} seconds'.format(clock_finish))
        else:
            print('Completed in {:.2f} seconds'.format(clock_finish))

def print_progress_bar(
        iteration,
        total,
        decimals=1,
        length=50,
):
    """ Call in a loop to create terminal progress bar

    Arguments:
        iteration: (Required) current iteration
        total:     (Required) total iterations
        decimals:  (Optional) positive number of decimals in percent complete
        length:    (Optional) character length of bar
    """
    progress = ("{0:." + str(decimals) + "f}").format(total - iteration)
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\rProgress:|{bar}| {progress}/{float(total)} simulated seconds remaining    ', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

# Tests
def test_recursive_store():
    environment_config = {
        'environment': {
            'temperature': {
                '_default': 0.0,
                '_updater': 'accumulate'},
            'fields': {
                (0, 1): {
                    'enzymeX': {
                        '_default': 0.0,
                        '_updater': 'set'},
                    'enzymeY': {
                        '_default': 0.0,
                        '_updater': 'set'}},
                (0, 2): {
                    'enzymeX': {
                        '_default': 0.0,
                        '_updater': 'set'},
                    'enzymeY': {
                        '_default': 0.0,
                        '_updater': 'set'}}},
            'agents': {
                '1': {
                    'location': {
                        '_default': (0, 0),
                        '_updater': 'set'},
                    'boundary': {
                        'external': {
                            '_default': 0.0,
                            '_updater': 'set'},
                        'internal': {
                            '_default': 0.0,
                            '_updater': 'set'}},
                    'transcripts': {
                        'flhDC': {
                            '_default': 0,
                            '_updater': 'accumulate'},
                        'fliA': {
                            '_default': 0,
                            '_updater': 'accumulate'}},
                    'proteins': {
                        'ribosome': {
                            '_default': 0,
                            '_updater': 'set'},
                        'flagella': {
                            '_default': 0,
                            '_updater': 'accumulate'}}},
                '2': {
                    'location': {
                        '_default': (0, 0),
                        '_updater': 'set'},
                    'boundary': {
                        'external': {
                            '_default': 0.0,
                            '_updater': 'set'},
                        'internal': {
                            '_default': 0.0,
                            '_updater': 'set'}},
                    'transcripts': {
                        'flhDC': {
                            '_default': 0,
                            '_updater': 'accumulate'},
                        'fliA': {
                            '_default': 0,
                            '_updater': 'accumulate'}},
                    'proteins': {
                        'ribosome': {
                            '_default': 0,
                            '_updater': 'set'},
                        'flagella': {
                            '_default': 0,
                            '_updater': 'accumulate'}}}}}}

    state = Store(environment_config)
    state.apply_update({})
    state.state_for(['environment'], ['temperature'])


def test_in():
    blank = {}
    path = ['where', 'are', 'we']
    assoc_path(blank, path, 5)
    print(blank)
    print(get_in(blank, path))
    blank = update_in(blank, path, lambda x: x + 6)
    print(blank)


quark_colors = ['green', 'red', 'blue']
quark_spins = ['up', 'down']
electron_spins = ['-1/2', '1/2']
electron_orbitals = [
    str(orbit) + 's'
    for orbit in range(1, 8)]


class Proton(Process):
    name = 'proton'
    defaults = {
        'time_step': 1.0,
        'radius': 0.0}

    def __init__(self, parameters=None):
        super(Proton, self).__init__(parameters)

    def ports_schema(self):
        return {
            'radius': {
                '_updater': 'set',
                '_default': self.parameters['radius']},
            'quarks': {
                '_divider': 'split_dict',
                '*': {
                    'color': {
                        '_updater': 'set',
                        '_default': quark_colors[0]},
                    'spin': {
                        '_updater': 'set',
                        '_default': quark_spins[0]}}},
            'electrons': {
                '*': {
                    'orbital': {
                        '_updater': 'set',
                        '_default': electron_orbitals[0]},
                    'spin': {
                        '_default': electron_spins[0]}}}}

    def next_update(self, timestep, states):
        update = {}

        collapse = np.random.random()
        if collapse < states['radius'] * timestep:
            update['radius'] = collapse
            update['quarks'] = {}

            for name, quark in states['quarks'].items():
                update['quarks'][name] = {
                    'color': np.random.choice(quark_colors),
                    'spin': np.random.choice(quark_spins)}

            update['electrons'] = {}
            orbitals = electron_orbitals.copy()
            for name, electron in states['electrons'].items():
                np.random.shuffle(orbitals)
                update['electrons'][name] = {
                    'orbital': orbitals.pop()}

        return update

class Electron(Process):
    name = 'electron'
    defaults = {
        'time_step': 1.0,
        'spin': electron_spins[0]}

    def __init__(self, parameters=None):
        super(Electron, self).__init__(parameters)

    def ports_schema(self):
        return {
            'spin': {
                '_updater': 'set',
                '_default': self.parameters['spin']},
            'proton': {
                'radius': {
                    '_default': 0.0}}}

    def next_update(self, timestep, states):
        update = {}

        if np.random.random() < states['proton']['radius']:
            update['spin'] = np.random.choice(electron_spins)

        return update


def make_proton(parallel=False):
    processes = {
        'proton': Proton({'_parallel': parallel}),
        'electrons': {
            'a': {
                'electron': Electron({'_parallel': parallel})},
            'b': {
                'electron': Electron()}}}

    spin_path = ('internal', 'spin')
    radius_path = ('structure', 'radius')

    topology = {
        'proton': {
            'radius': radius_path,
            'quarks': ('internal', 'quarks'),
            'electrons': {
                '_path': ('electrons',),
                '*': {
                    'orbital': ('shell', 'orbital'),
                    'spin': spin_path}}},
        'electrons': {
            'a': {
                'electron': {
                    'spin': spin_path,
                    'proton': {
                        '_path': ('..', '..'),
                        'radius': radius_path}}},
            'b': {
                'electron': {
                    'spin': spin_path,
                    'proton': {
                        '_path': ('..', '..'),
                        'radius': radius_path}}}}}

    initial_state = {
        'structure': {
            'radius': 0.7},
        'internal': {
            'quarks': {
                'x': {
                    'color': 'green',
                    'spin': 'up'},
                'y': {
                    'color': 'red',
                    'spin': 'up'},
                'z': {
                    'color': 'blue',
                    'spin': 'down'}}}}

    return {
        'processes': processes,
        'topology': topology,
        'initial_state': initial_state}


class Sine(Process):
    name = 'sine'
    defaults = {
        'initial_phase': 0.0}

    def __init__(self, parameters=None):
        super(Sine, self).__init__(parameters)

    def ports_schema(self):
        return {
            'frequency': {
                '_default': 440.0},
            'amplitude': {
                '_default': 1.0},
            'phase': {
                '_default': self.parameters['initial_phase']},
            'signal': {
                '_default': 0.0,
                '_updater': 'set'}}

    def next_update(self, timestep, states):
        phase_shift = timestep * states['frequency'] % 1.0
        signal = states['amplitude'] * math.sin(
            2 * math.pi * (states['phase'] + phase_shift))

        return {
            'phase': phase_shift,
            'signal': signal}


def test_topology_ports():
    proton = make_proton()

    experiment = Experiment(proton)

    log.debug(pf(experiment.state.get_config(True)))

    experiment.update(10.0)

    log.debug(pf(experiment.state.get_config(True)))
    log.debug(pf(experiment.state.divide_value()))


def test_timescales():
    class Slow(Process):
        name = 'slow'
        defaults = {'timestep': 3.0}

        def __init__(self, config=None):
            super(Slow, self).__init__(config)

        def ports_schema(self):
            return {
                'state': {
                    'base': {
                        '_default': 1.0}}}

        def local_timestep(self):
            return self.parameters['timestep']

        def next_update(self, timestep, states):
            base = states['state']['base']
            next_base = timestep * base * 0.1

            return {
                'state': {'base': next_base}}

    class Fast(Process):
        name = 'fast'
        defaults = {'timestep': 0.3}

        def __init__(self, config=None):
            super(Fast, self).__init__(config)

        def ports_schema(self):
            return {
                'state': {
                    'base': {
                        '_default': 1.0},
                    'motion': {
                        '_default': 0.0}}}

        def local_timestep(self):
            return self.parameters['timestep']

        def next_update(self, timestep, states):
            base = states['state']['base']
            motion = timestep * base * 0.001

            return {
                'state': {'motion': motion}}

    processes = {
        'slow': Slow(),
        'fast': Fast()}

    states = {
        'state': {
            'base': 1.0,
            'motion': 0.0}}

    topology = {
        'slow': {'state': ('state',)},
        'fast': {'state': ('state',)}}

    emitter = {'type': 'null'}
    experiment = Experiment({
        'processes': processes,
        'topology': topology,
        'emitter': emitter,
        'initial_state': states})

    experiment.update(10.0)


def test_inverse_topology():
    update = {
        'port1': {
            'a': 5},
        'port2': {
            'b': 10},
        'port3': {
            'b': 10},
        'global': {
            'c': 20}}

    topology = {
        'port1': ('boundary', 'x'),
        'global': ('boundary',),
        'port2': ('boundary', 'y'),
        'port3': ('boundary', 'x')}

    path = ('agent',)
    inverse = inverse_topology(path, update, topology)
    expected_inverse = {
        'agent': {
            'boundary': {
                'x': {
                    'a': 5,
                    'b': 10},
                'y': {
                    'b': 10},
                'c': 20}}}

    assert inverse == expected_inverse

def test_2_store_1_port():
    """
    Split one port of a processes into two stores
    """
    class OnePort(Process):
        name = 'one_port'
        def ports_schema(self):
            return {
                'A': {
                    'a': {
                        '_default': 0,
                        '_emit': True},
                    'b': {
                        '_default': 0,
                        '_emit': True}
                }
            }
        def next_update(self, timestep, states):
            return {
                'A': {
                    'a': 1,
                    'b': 2}}

    class SplitPort(Generator):
        """splits OnePort's ports into two stores"""
        name = 'split_port_generator'
        def generate_processes(self, config):
            return {
                'one_port': OnePort({})}
        def generate_topology(self, config):
            return {
                'one_port': {
                    'A': {
                        'a': ('internal', 'a',),
                        'b': ('external', 'a',)
                    }
                }}

    # run experiment
    split_port = SplitPort({})
    network = split_port.generate()
    exp = Experiment({
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'external': {'a': [0, 2, 4]},
        'internal': {'a': [0, 1, 2]},
        'time': [0.0, 1.0, 2.0]}
    assert output == expected_output


def test_multi_port_merge():
    class MultiPort(Process):
        name = 'multi_port'
        def ports_schema(self):
            return {
                'A': {
                    'a': {
                        '_default': 0,
                        '_emit': True}},
                'B': {
                    'a': {
                        '_default': 0,
                        '_emit': True}},
                'C': {
                    'a': {
                        '_default': 0,
                        '_emit': True}}}
        def next_update(self, timestep, states):
            return {
                'A': {'a': 1},
                'B': {'a': 1},
                'C': {'a': 1}}

    class MergePort(Generator):
        """combines both of MultiPort's ports into one store"""
        name = 'multi_port_generator'
        def generate_processes(self, config):
            return {
                'multi_port': MultiPort({})}
        def generate_topology(self, config):
            return {
                'multi_port': {
                    'A': ('aaa',),
                    'B': ('aaa',),
                    'C': ('aaa',)}}

    # run experiment
    merge_port = MergePort({})
    network = merge_port.generate()
    exp = Experiment({
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'aaa': {'a': [0, 3, 6]},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


def test_complex_topology():
    class Po(Process):
        name = 'po'

        def ports_schema(self):
            return {
                'A': {
                    'a': {'_default': 0},
                    'b': {'_default': 0},
                    'c': {'_default': 0}},
                'B': {
                    'd': {'_default': 0},
                    'e': {'_default': 0}}}

        def next_update(self, timestep, states):
            return {
                'A': {
                    'a': states['A']['b'],
                    'b': states['A']['c'],
                    'c': states['B']['d'] + states['B']['e']},
                'B': {
                    'd': states['A']['a'],
                    'e': states['B']['e']}}

    class Qo(Process):
        name = 'qo'

        def ports_schema(self):
            return {
                'D': {
                    'x': {'_default': 0},
                    'y': {'_default': 0},
                    'z': {'_default': 0}},
                'E': {
                    'u': {'_default': 0},
                    'v': {'_default': 0}}}

        def next_update(self, timestep, states):
            return {
                'D': {
                    'x': -1,
                    'y': 12,
                    'z': states['D']['x'] + states['D']['y']},
                'E': {
                    'u': 3,
                    'v': states['E']['u']}}


    class PoQo(Generator):
        def generate_processes(self, config=None):
            P = Po(config)
            Q = Qo(config)

            return {
                'po': P,
                'qo': Q}

        def generate_topology(self, config=None):
            return {
                'po': {
                    'A': {
                        '_path': ('aaa',),
                        'b': ('o',)},
                    'B': ('bbb',)},
                'qo': {
                    'D': {
                        'x': ('aaa', 'a'),
                        'y': ('aaa', 'o'),
                        'z': ('ddd', 'z')},
                    'E': {
                        'u': ('aaa', 'u'),
                        'v': ('bbb', 'e')}}}


    initial_state = {
        'aaa': {
            'a': 2,
            'c': 5,
            'o': 3,
            'u': 11},
        'bbb': {
            'd': 14,
            'e': 88},
        'ddd': {
            'z': 333}}

    PQ = PoQo({})
    pq_config = PQ.generate()
    pq_config['initial_state'] = initial_state

    experiment = Experiment(pq_config)

    pp(experiment.state.get_value())
    experiment.update(1)

    state = experiment.state.get_value()
    assert state['aaa']['a'] == initial_state['aaa']['a'] + initial_state['aaa']['o'] - 1
    assert state['aaa']['o'] == initial_state['aaa']['o'] + initial_state['aaa']['c'] + 12
    assert state['aaa']['c'] == initial_state['aaa']['c'] + initial_state['bbb']['d'] + initial_state['bbb']['e']
    assert state['aaa']['u'] == initial_state['aaa']['u'] + 3
    assert state['bbb']['d'] == initial_state['bbb']['d'] + initial_state['aaa']['a']
    assert state['bbb']['e'] == initial_state['bbb']['e'] + initial_state['bbb']['e'] + initial_state['aaa']['u']
    assert state['ddd']['z'] == initial_state['ddd']['z'] + initial_state['aaa']['a'] + initial_state['aaa']['o']


def test_multi():
    with Pool(processes=4) as pool:
        multi = MultiInvoke(pool)
        proton = make_proton()
        experiment = Experiment({**proton, 'invoke': multi.invoke})

        log.debug(pf(experiment.state.get_config(True)))

        experiment.update(10.0)

        log.debug(pf(experiment.state.get_config(True)))
        log.debug(pf(experiment.state.divide_value()))


def test_parallel():
    proton = make_proton(parallel=True)
    experiment = Experiment(proton)

    log.debug(pf(experiment.state.get_config(True)))

    experiment.update(10.0)

    log.debug(pf(experiment.state.get_config(True)))
    log.debug(pf(experiment.state.divide_value()))

    experiment.end()


class TestUpdateIn:
    d = {
        'foo': {
            1: {
                'a': 'b',
            },
        },
        'bar': {
            'c': 'd',
        },
    }

    def test_simple(self):
        updated = copy.deepcopy(self.d)
        updated = update_in(
            updated, ('foo', 1, 'a'), lambda current: 'updated')
        expected = {
            'foo': {
                1: {
                    'a': 'updated',
                },
            },
            'bar': {
                'c': 'd',
            },
        }
        assert updated == expected

    def test_add_leaf(self):
        updated = copy.deepcopy(self.d)
        updated = update_in(
            updated, ('foo', 1, 'new'), lambda current: 'updated')
        expected = {
            'foo': {
                1: {
                    'a': 'b',
                    'new': 'updated',
                },
            },
            'bar': {
                'c': 'd',
            },
        }
        assert updated == expected

    def test_add_dict(self):
        updated = copy.deepcopy(self.d)
        updated = update_in(
            updated, ('foo', 2), lambda current: {'a': 'updated'})
        expected = {
            'foo': {
                1: {
                    'a': 'b',
                },
                2: {
                    'a': 'updated',
                },
            },
            'bar': {
                'c': 'd',
            },
        }
        assert updated == expected

    def test_complex_merge(self):
        updated = copy.deepcopy(self.d)
        updated = update_in(
            updated, ('foo',),
            lambda current: deep_merge(
                current,
                {'foo': {'a': 'updated'}, 'b': 2}),
            )
        expected = {
            'foo': {
                'foo': {
                    'a': 'updated',
                },
                'b': 2,
                1: {
                    'a': 'b',
                },
            },
            'bar': {
                'c': 'd',
            },
        }
        assert updated == expected

    def test_add_to_root(self):
        updated = copy.deepcopy(self.d)
        updated = update_in(
            updated,
            tuple(),
            lambda current: deep_merge(current, ({'a': 'updated'})),
        )
        expected = {
            'foo': {
                1: {
                    'a': 'b',
                },
            },
            'bar': {
                'c': 'd',
            },
            'a': 'updated'
        }
        assert updated == expected

    def test_set_root(self):
        updated = copy.deepcopy(self.d)
        updated = update_in(
            updated, tuple(), lambda current: {'a': 'updated'})
        expected = {
            'a': 'updated',
        }
        assert updated == expected


def test_depth():
    nested = {
        'A': {
            'AA': 5,
            'AB': {
                'ABC': 11}},
        'B': {
            'BA': 6}}

    dissected = depth(nested)
    assert len(dissected) == 3
    assert dissected[('A', 'AB', 'ABC')] == 11


def test_deletion():
    nested = {
        'A': {
            'AA': 5,
            'AB': {
                'ABC': 11}},
        'B': {
            'BA': 6}}

    delete_in(nested, ('A', 'AA'))
    assert 'AA' not in nested['A']


def test_sine():
    sine = Sine()
    print(sine.next_update(0.25 / 440.0, {
        'frequency': 440.0,
        'amplitude': 0.1,
        'phase': 1.5}))


if __name__ == '__main__':
    # test_recursive_store()
    # test_in()
    # test_timescales()
    # test_topology_ports()
    # test_multi()
    # test_sine()
    # test_parallel()
    # test_complex_topology()
    # test_multi_port_merge()

    test_2_store_1_port()
