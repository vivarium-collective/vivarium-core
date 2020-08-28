'''
==========================================
Experiment and Store Classes
==========================================
'''


from __future__ import absolute_import, division, print_function

import os
import copy
import math
import random
import datetime

import numpy as np
import logging as log

import pprint
pretty=pprint.PrettyPrinter(indent=2)

def pp(x):
    pretty.pprint(x)

def pf(x):
    return pretty.pformat(x)

from multiprocessing import Pool

from vivarium.library.units import Quantity
from vivarium.library.dict_utils import merge_dicts, deep_merge, deep_merge_check
from vivarium.core.emitter import get_emitter
from vivarium.core.process import (
    Process,
    ParallelProcess,
    serialize_dictionary,
)
from vivarium.core.registry import (
    divider_registry,
    updater_registry,
    serializer_registry,
)


INFINITY = float('inf')
VERBOSE = False

log.basicConfig(level=os.environ.get("LOGLEVEL", log.WARNING))


# Store
def key_for_value(d, looking):
    found = None
    for key, value in d.items():
        if looking == value:
            found = key
            break
    return found


def get_in(d, path, default=None):
    if path:
        head = path[0]
        if head in d:
            return get_in(d[head], path[1:])
        return default
    return d


def assoc_path(d, path, value):
    if path:
        head = path[0]
        if len(path) == 1:
            d[head] = value
        else:
            if head not in d:
                d[head] = {}
            assoc_path(d[head], path[1:], value)
    else:
        value


def update_in(d, path, f):
    if path:
        head = path[0]
        d.setdefault(head, {})
        updated = copy.deepcopy(d)
        updated[head] = update_in(d[head], path[1:], f)
        return updated
    return f(d)


def dissoc(d, removing):
    return {
        key: value
        for key, value in d.items()
        if key not in removing}


def without(d, removing):
    return {
        key: value
        for key, value in d.items()
        if key != removing}

def schema_for(port, keys, initial_state, default=0.0, updater='accumulate'):
    return {
        key: {
            '_default': initial_state.get(
                port, {}).get(key, default),
            '_updater': updater}
        for key in keys}


def always_true(x):
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
    schema_keys = set([
        '_default',
        '_updater',
        '_value',
        '_properties',
        '_emit',
        '_serializer',
    ])

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
        self.deleted = False
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
            as well. You can also provide your own function here instead of
            a string key into the updater library.
        * _emit - whether or not to emit the values under this point in the tree.
        * _divider - What to do with this node when division happens.
            Default behavior is to leave it alone, but you can also pass
            'split' here, or a function of your choosing. If you need other
            values from the state you need to supply a dictionary here
            containing the updater and the topology for where the other
            state values are coming from. This has two keys:
            * divider - a function that takes the existing value and any
                values supplied from the adjoining topology.
            * topology - a mapping of keys to paths where the value for
                those keys will be found. This will be passed in as the second
                argument to the divider function.
        * _subschema/* - If this node was declared to house an unbounded set
            of related states, the schema for these states is held in this
            nodes subschema and applied whenever new subkeys are added here.
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

    def mark_deleted(self):
        '''
        When nodes are removed from the tree, they are marked as deleted
        in case something else has a reference to them.
        '''

        self.deleted = True
        if self.inner:
            for child in self.inner.values():
                child.mark_deleted()

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
                lost.mark_deleted()
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

    def reduce_to(self, path, reducer, initial=None):
        value = self.reduce(reducer, initial)
        assoc_path({}, path, value)
        self.apply_update(update)

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
        * `_delete` - The value here is a list of paths to delete from
            the tree.
        * `_add` - Adds a state into the subtree:
            * path - Path to the added state key.
            * state - The value of the added state.
        * `_generate` - The value has four keys, which are essentially
            the arguments to the `generate()` function:
            * path - Path into the tree to generate this subtree.
            * processes - Tree of processes to generate.
            * topology - Connections of all the process's `ports_schema()`.
            * initial_state - Initial state for this new subtree.
        * `_divide` - Performs cell division by constructing two new
            daugther cells and removing the mother. Takes a dict with two keys:
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
                The result of the reduction will be assigned to this
                point in the tree.
        '''

        if self.inner or self.subschema:
            topology_updates = {}

            if '_delete' in update:
                # delete a list of paths
                for path in update['_delete']:
                    self.delete_path(path)

                update = dissoc(update, ['_delete'])

            if '_add' in update:
                # add a list of sub-compartments
                for added in update['_add']:
                    path = added['path']
                    state = added['state']
                    target = self.establish_path(path, {})
                    self.apply_subschemas()
                    self.apply_defaults()
                    target.set_value(state)

                update = dissoc(update, ['_add'])

            if '_generate' in update:
                # generate a list of new compartments
                for generate in update['_generate']:
                    self.generate(
                        generate['path'],
                        generate['processes'],
                        generate['topology'],
                        generate['initial_state'])
                    assoc_path(
                        topology_updates,
                        generate['path'],
                        generate['topology'])
                self.apply_subschemas()
                self.apply_defaults()

                update = dissoc(update, '_generate')

            if '_divide' in update:
                # use dividers to find initial states for daughters
                divide = update['_divide']
                mother = divide['mother']
                daughters = divide['daughters']
                initial_state = self.inner[mother].get_value(
                    condition=lambda child: not (isinstance(child.value, Process)),
                    f=lambda child: copy.deepcopy(child))
                states = self.inner[mother].divide_value()

                for daughter, state in zip(daughters, states):
                    daughter_id = daughter['daughter']

                    # use initiapl state as default, merge in divided values
                    initial_state = deep_merge(
                        initial_state,
                        state)

                    self.generate(
                        daughter['path'],
                        daughter['processes'],
                        daughter['topology'],
                        daughter['initial_state'])
                    assoc_path(
                        topology_updates,
                        daughter['path'],
                        daughter['topology'])

                    self.apply_subschemas()
                    self.inner[daughter_id].set_value(initial_state)
                    self.apply_defaults()
                self.delete_path((mother,))

                update = dissoc(update, '_divide')

            for key, value in update.items():
                if key in self.inner:
                    child = self.inner[key]
                    inner_updates = child.apply_update(
                        value, process_topology, state)
                    if inner_updates:
                        topology_updates = deep_merge(
                            topology_updates,
                            {key: inner_updates})
                # elif self.subschema:
                #     self.inner[key] = Store(self.subschema, self)
                #     self.inner[key].set_value(value)
                #     self.inner[key].apply_defaults()

            return topology_updates

        else:
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

            value = self.value
            if port_mapping is not None:
                updater_topology = {
                    updater_port: process_topology[proc_port]
                    for updater_port, proc_port in port_mapping.items()
                }
                state_dict = state.outer.topology_state(
                    updater_topology)

            self.value = updater(self.value, update, state_dict)

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
                    state[key] = node.schema_topology(subschema, {})

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
                    'topology at path {} and source {} has keys that are not in the schema: {}'.format(
                        self.path_for(), source, mismatch_topology))
            if mismatch_schema:
                log.debug('{} schema has keys not in topology: {}'.format(
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


def inverse_topology(outer, update, topology):
    '''
    Transform an update from the form its process produced into
    one aligned to the given topology.

    The inverse of this function (using a topology to construct a view for
    the perspective of a Process ports_schema()) lives in `Store`, called
    `topology_state`. This one stands alone as it does not require a store
    to calculate.
    '''

    inverse = {}
    for key, path in topology.items():

        if key == '*':

            if isinstance(path, dict):
                node = inverse
                if '_path' in path:
                    inner = normalize_path(outer + path['_path'])
                    node = get_in(inverse, inner)
                    if node is None:
                        node = {}
                        assoc_path(inverse, inner, node)
                    path = without(path, '_path')

                for child, child_update in update.items():
                    node[child] = inverse_topology(
                        tuple(),
                        update[child],
                        path)

            else:
                for child, child_update in update.items():
                    inner = normalize_path(outer + path + (child,))
                    if isinstance(child_update, dict):
                        inverse = update_in(
                            inverse,
                            inner,
                            lambda current: deep_merge(
                                current, child_update),
                        )
                    else:
                        assoc_path(inverse, inner, child_update)

        elif key in update:
            value = update[key]
            if isinstance(path, dict):
                node = inverse
                if '_path' in path:
                    inner = normalize_path(outer + path['_path'])
                    node = get_in(inverse, inner)
                    if node is None:
                        node = {}
                        assoc_path(inverse, inner, node)
                    path = without(path, '_path')

                node.update(inverse_topology(
                    tuple(),
                    value,
                    path))

            else:
                inner = normalize_path(outer + path)
                if isinstance(value, dict):
                    inverse = update_in(
                        inverse,
                        inner,
                        lambda current: deep_merge(current, value)
                    )
                else:
                    assoc_path(inverse, inner, value)

    return inverse


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


def normalize_path(path):
    progress = []
    for step in path:
        if step == '..' and len(progress) > 0:
            progress = progress[:-1]
        else:
            progress.append(step)
    return progress


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
        """Defines simulations

        Arguments:
            config (dict): A dictionary of configuration options. The
                required options are:

                * **processes** (:py:class:`dict`): A dictionary that
                    maps :term:`process` names to process objects. You
                    will usually get this from the ``processes``
                    attribute of the dictionary from
                    :py:meth:`vivarium.core.experiment.Compartment.generate`.
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
            'experiment_id', timestamp(datetime.datetime.utcnow()))
        self.experiment_name = config.get('experiment_name', self.experiment_id)
        self.description = config.get('description', '')
        self.processes = config['processes']
        self.topology = config['topology']
        self.initial_state = config.get('initial_state', {})
        self.emit_step = config.get('emit_step')

        self.invoke = config.get('invoke', InvokeProcess)
        self.parallel = {}

        self.state = generate_state(
            self.processes,
            self.topology,
            self.initial_state)

        emitter_config = config.get('emitter', {})
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

        log.info('experiment {}'.format(self.experiment_id))

        log.info('\nPROCESSES:')
        log.info(pf(self.processes))

        log.info('\nTOPOLOGY:')
        log.info(pf(self.topology))

        log.info('\nSTATE:')
        log.info(pf(self.state.get_value()))

        log.info('\nCONFIG:')
        log.info(pf(self.state.get_config(True)))

    def emit_configuration(self):
        data = {
            'time_created': timestamp(),
            'experiment_id': self.experiment_id,
            'name': self.experiment_name,
            'description': self.description,
            'processes': serialize_dictionary(self.processes),
            'topology': self.topology,
            # 'state': self.state.get_config()
        }
        emit_config = {
            'table': 'configuration',
            'data': data}
        self.emitter.emit(emit_config)

    def invoke_process(self, process, path, interval, states):
        if process.parallel:
            # add parallel process if it doesn't exist
            if not path in self.parallel:
                self.parallel[path] = ParallelProcess(process)
            # trigger the computation of the parallel process
            self.parallel[path].update(interval, states)

            return self.parallel[path]
        else:
            # if not parallel, perform a normal invocation
            return self.invoke(process, interval, states)

    def process_update(self, path, state, interval):
        process = state.value
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
        topology_updates = self.state.apply_update(
            update, process_topology, state)
        if topology_updates:
            self.topology = deep_merge(self.topology, topology_updates)

    def run_derivers(self, derivers):
        updates = []
        for path, deriver in derivers.items():
            # timestep shouldn't influence derivers
            if not deriver.deleted:
                update, process_topology, state = self.process_update(
                    path, deriver, 0)
                self.apply_update(update.get(), process_topology, state)

    def emit_data(self):
        data = self.state.emit_data()
        data.update({
            'time': self.local_time})
        emit_config = {
            'table': 'history',
            'data': serialize_dictionary(data),
        }
        self.emitter.emit(emit_config)

    def send_updates(self, update_tuples, derivers=None):
        for update_tuple in update_tuples:
            update, process_topology, state = update_tuple
            self.apply_update(update.get(), process_topology, state)

        if derivers is None:
            derivers = {
                path: state
                for path, state in self.state.depth()
                if state.value is not None and isinstance(state.value, Process) and state.value.is_deriver()}

        self.run_derivers(derivers)

    def update(self, interval):
        """ Run each process for the given interval and update the related states. """

        time = 0
        emit_time = self.emit_step

        def empty_front(t):
            return {
                'time': t,
                'update': {}}

        # keep track of which processes have simulated until when
        front = {}

        while time < interval:
            full_step = INFINITY

            if VERBOSE:
                for state_id in self.states:
                    print('{}: {}'.format(time, self.states[state_id].to_dict()))

            # find all existing processes and derivers in the tree
            processes = {}
            derivers = {}
            for path, state in self.state.depth():
                if state.value is not None and isinstance(state.value, Process):
                    if state.value.is_deriver():
                        derivers[path] = state
                    else:
                        processes[path] = state

            # find any parallel processes that were removed and terminate them
            for terminated in self.parallel.keys() - processes.keys():
                self.parallel[terminated].end()
                del self.parallel[terminated]

            # setup a way to track how far each process has simulated in time
            front = {
                path: process
                for path, process in front.items()
                if path in processes}

            # go through each process and find those that are able to update
            # based on their current time being less than the global time.
            for path, state in processes.items():
                if not path in front:
                    front[path] = empty_front(time)
                process_time = front[path]['time']

                if process_time <= time:
                    process = state.value
                    future = min(process_time + process.local_timestep(), interval)
                    timestep = future - process_time

                    # calculate the update for this process
                    update = self.process_update(path, state, timestep)

                    # store the update to apply at its projected time
                    if timestep < full_step:
                        full_step = timestep
                    front[path]['time'] = future
                    front[path]['update'] = update

            if full_step == INFINITY:
                # no processes ran, jump to next process
                next_event = interval
                for process_name in front.keys():
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

                self.send_updates(updates, derivers)

                time = future
                self.local_time += full_step
                log.info('time: {}'.format(self.local_time))

                if self.emit_step is None:
                    self.emit_data()
                elif emit_time <= time:
                    while emit_time <= time:
                        self.emit_data()
                        emit_time += self.emit_step

        for process_name, advance in front.items():
            assert advance['time'] == time == interval
            assert len(advance['update']) == 0


    def end(self):
        for parallel in self.parallel.values():
            parallel.end()


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
            'a': 5,
        },
        'port2': {
            'b': 10,
        },
        'port3': {
            'b': 10,
        },
        'global': {
            'c': 20,
        },
    }
    topology = {
        'port1': ('boundary', 'x'),
        'global': ('boundary',),
        'port2': ('boundary', 'y'),
        'port3': ('boundary', 'x'),
    }
    path = ('agent',)
    inverse = inverse_topology(path, update, topology)
    expected_inverse = {
        'agent': {
            'boundary': {
                'x': {
                    'a': 5,
                    'b': 10,
                },
                'y': {
                    'b': 10,
                },
                'c': 20,
            }
        }
    }
    assert inverse == expected_inverse


def test_multi():
    with Pool(processes=4) as pool:
        multi = MultiInvoke(pool)
        proton = make_proton()
        experiment = Experiment(dict(
            proton,
            invoke=multi.invoke))

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

    test_parallel()
