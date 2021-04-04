import copy
import re

from vivarium.library.dict_utils import (
    deep_merge, deep_merge_multi_update)
from vivarium.core.types import (
    HierarchyPath, TuplePath, Topology)


def get_in(d, path, default=None):
    if path:
        head = path[0]
        if head in d:
            return get_in(d[head], path[1:])
        return default
    return d


def delete_in(d, path):
    if len(path) > 0:
        head = path[0]
        if len(path) == 1:
            # at the node to be deleted
            if head in d:
                del d[head]
        elif head in d:
            # down = d[head]
            delete_in(d[head], path[1:])


def assoc_path(d, path, value):
    if path:
        head = path[0]
        if len(path) == 1:
            d[head] = value
        else:
            if head not in d:
                d[head] = {}
            assoc_path(d[head], path[1:], value)
    elif isinstance(value, dict):
        deep_merge(d, value)


def without(d, removing):
    return {
        key: value
        for key, value in d.items()
        if key != removing}


def update_in(d, path, f):
    if path:
        head = path[0]
        d.setdefault(head, {})
        updated = copy.deepcopy(d)
        updated[head] = update_in(d[head], path[1:], f)
        return updated
    return f(d)


def paths_to_dict(path_list, f=lambda x: x):
    d = {}
    for path, node in path_list:
        assoc_path(d, path, f(node))
    return d


def dict_to_paths(root, d):
    if isinstance(d, dict):
        deeper = []
        for key, down in d.items():
            paths = dict_to_paths(root + (key,), down)
            deeper.extend(paths)
        return deeper
    else:
        return [(root, d)]
        

def inverse_topology(outer, update, topology, inverse=None):
    '''
    Transform an update from the form its process produced into
    one aligned to the given topology.

    The inverse of this function (using a topology to construct a view for
    the perspective of a Process ports_schema()) lives in `Store`, called
    `topology_state`. This one stands alone as it does not require a store
    to calculate.
    '''

    inverse = inverse or {}

    for key, path in topology.items():
        if key == '*':

            if isinstance(path, dict):
                node = inverse
                if '_path' in path:
                    inner = normalize_path(outer + path['_path'])
                    path = without(path, '_path')
                else:
                    inner = outer

                for child, child_update in update.items():
                    inverse = inverse_topology(
                        inner + (child,),
                        update[child],
                        path,
                        inverse)
            else:
                for child, child_update in update.items():
                    inner = normalize_path(outer + path + (child,))
                    if isinstance(child_update, dict):
                        inverse = update_in(
                            inverse,
                            inner,
                            lambda current: deep_merge(
                                current, child_update))
                    else:
                        assoc_path(inverse, inner, child_update)

        elif key in update:
            value = update[key]
            if isinstance(path, dict):
                if '_path' in path:
                    inner = normalize_path(outer + path['_path'])
                    path = without(path, '_path')

                    for update_key in update[key].keys():
                        if update_key not in path:
                            path[update_key] = (update_key,)
                else:
                    inner = outer

                inverse = inverse_topology(
                    inner,
                    value,
                    path,
                    inverse)
            else:
                inner = normalize_path(outer + path)
                if isinstance(value, dict):
                    inverse = update_in(
                        inverse,
                        inner,
                        lambda current: deep_merge_multi_update(current, value))
                else:
                    assoc_path(inverse, inner, value)
    return inverse


def normalize_path(path):
    progress = []
    for step in path:
        if step == '..' and len(progress) > 0:
            progress = progress[:-1]
        else:
            progress.append(step)
    return tuple(progress)


_PATH_ELEMENT = re.compile(r'[^>/]+')

def convert_path_to_tuple(path: HierarchyPath) -> TuplePath:
    """ convert paths to tuple format """
    if isinstance(path, str):
        path = tuple(_PATH_ELEMENT.findall(path.replace('<', '>..>')))
    return path


def convert_topology_path(topology: Topology) -> Topology:
    """ convert a topology's paths to tuple format """
    converted_topology: Topology = {}
    for name, path in topology.items():
        if isinstance(path, dict):
            converted_topology[name] = convert_topology_path(path)
        else:
            converted_topology[name] = convert_path_to_tuple(path)
    return converted_topology


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


def test_in():
    blank = {}
    path = ['where', 'are', 'we']
    assoc_path(blank, path, 5)
    print(blank)
    print(get_in(blank, path))
    blank = update_in(blank, path, lambda x: x + 6)
    print(blank)

def test_path_declare():

    path_down = 'path>to>store'
    new_path_down = convert_path_to_tuple(path_down)
    assert new_path_down == ('path', 'to', 'store')

    path_up = '<<store'
    new_path_up = convert_path_to_tuple(path_up)
    assert new_path_up == ('..', '..', 'store')

def test_topology_paths():

    # complex topology test
    topology = {
        'a': {
            '1': 'path>to>A',
            '2': 'path>to>B',
            '3': '<<C'},
        'b': {
            '1': 'path>to>B',
            '2': 'path>to>A'},
        'c': {
            'cc': {
                '1': '<<path>to>B'}}}
    new_topology = convert_topology_path(topology)
    assert new_topology == {
        'a': {
            '1': ('path', 'to', 'A'),
            '2': ('path', 'to', 'B'),
            '3': ('..', '..', 'C')},
        'b': {
            '1': ('path', 'to', 'B'),
            '2': ('path', 'to', 'A')},
        'c': {
            'cc': {
                '1': ('..', '..', 'path', 'to', 'B')}}}

    # slash test
    topology = {'a': {'1': '../../path/to/A'}}
    new_topology = convert_topology_path(topology)
    assert new_topology == {'a': {'1': ('..', '..', 'path', 'to', 'A')}}

    # tuple test
    topology = {'a': {'1': ('path', 'to', 'A')}}
    new_topology = convert_topology_path(topology)
    assert new_topology == {'a': {'1': ('path', 'to', 'A')}}

    # boundary case test
    topology = {'a': {'1': ''}}
    new_topology = convert_topology_path(topology)
    assert new_topology == {'a': {'1': ()}}


if __name__ == '__main__':
    test_path_declare()
    test_topology_paths()
