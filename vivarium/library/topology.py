'''
==================
Topology Utilities
==================
'''

import copy
import re

from vivarium.library.dict_utils import deep_merge, deep_merge_multi_update


def get_in(d, path, default=None):
    '''Get the value from a dictionary by its path.

    >>> d = {'a': {'b': 'c', 'd': 'e'}}
    >>> get_in(d, ('a', 'b'))
    'c'
    >>> get_in(d, ('a', 'z'))
    >>> get_in(d, ('a', 'z'), 'y')
    'y'
    '''
    if path:
        head = path[0]
        if head in d:
            return get_in(d[head], path[1:], default)
        return default
    return d


def delete_in(d, path):
    '''Delete an item from a dictionary by its path.

    >>> d = {'a': {'b': 'c', 'd': 'e'}}
    >>> delete_in(d, ('a', 'b'))
    >>> d
    {'a': {'d': 'e'}}
    '''
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
    '''Insert ``value`` into the dictionary ``d`` at ``path``.

    >>> d = {'a': {'b': 'c'}}
    >>> assoc_path(d, ('a', 'd'), 'e')
    {'a': {'b': 'c', 'd': 'e'}}
    >>> d
    {'a': {'b': 'c', 'd': 'e'}}

    Create new dictionaries recursively as needed.
    '''

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
    return d


def without(d, removing):
    '''Get a copy of ``d`` without the keys in ``removing``.'''
    return {
        key: value
        for key, value in d.items()
        if key != removing}


def update_in(d, path, f):
    '''Update every value in a dictionary based on ``f``.

    Args:
        d: The dictionary ``path`` applies to. This object is not
            modified.
        path: Path to the sub-dictionary within ``d`` that should be
            updated.
        f: Function to call on every value in the dictionary to update.
            The updated dictionary's values will be return values from
            ``f``.

    Returns:
        A copy of ``d`` with all the values under ``path`` updated to
        the value returned when ``f`` is called on the original value.
    '''
    if path:
        head = path[0]
        d.setdefault(head, {})
        updated = copy.deepcopy(d)
        updated[head] = update_in(d[head], path[1:], f)
        return updated
    return f(d)


def paths_to_dict(path_list, f=lambda x: x):
    '''Create a new dictionary that has the paths in ``path_list``.

    Args:
        path_list: A list of tuples ``(path, value)``.
        f: A function to apply to each value before inserting it into
            the dictionary.

    Returns:
        A new dictionary with the specified values (after being passed
        through ``f``) at each associated path.
    '''
    d = {}
    for path, node in path_list:
        assoc_path(d, path, f(node))
    return d


def dict_to_paths(root, d):
    """Get all the paths in a dictionary.

    For example:

    >>> root = ('root', 'subroot')
    >>> d = {
    ...     'a': {
    ...         'b': 'c',
    ...     },
    ...     'd': 'e',
    ... }
    >>> dict_to_paths(root, d)
    [(('root', 'subroot', 'a', 'b'), 'c'), (('root', 'subroot', 'd'), 'e')]
    """
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
    """Make a path absolute by resolving ``..`` elements."""
    progress = []
    for step in path:
        if step == '..' and len(progress) > 0:
            progress = progress[:-1]
        else:
            progress.append(step)
    return tuple(progress)


def convert_path_style(path):
    if isinstance(path, str):
        path = re.sub(r'<', '..<', path)
        path = tuple(re.split('<|>', path))

    return path


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
    new_path_down = convert_path_style(path_down)
    assert new_path_down == ('path', 'to', 'store')

    path_up = '<<store'
    new_path_up = convert_path_style(path_up)
    assert new_path_up == ('..', '..', 'store')


if __name__ == '__main__':
    test_path_declare()
