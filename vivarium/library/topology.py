'''
==================
Topology Utilities
==================
'''

import collections.abc
import copy
import re

from vivarium.library.dict_utils import deep_merge


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
        # updated = copy.deepcopy(d)
        updated = copy.copy(d)
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

    Updates that overlap on the same store are merged with
    ``deep_merge``. When two contributions target the exact same
    leaf, the last one wins; callers that need both applied (e.g.
    :py:func:`vivarium.core.engine.invert_topology`) split the update
    into separate, sequentially-applied updates before calling this.
    '''

    inverse = inverse or {}

    for key, path in topology.items():
        if key == '*':
            # keys with their own route are not glob children
            glob_children = [
                (child, child_update)
                for child, child_update in update.items()
                if child == '*' or child not in topology]
            if isinstance(path, dict):
                if '_path' in path:
                    path = path.copy()
                    inner = normalize_path(outer + path.pop('_path'))
                else:
                    inner = outer

                for child, child_update in glob_children:
                    inverse = inverse_topology(
                        inner + (child,),
                        child_update,
                        path,
                        inverse)
            else:
                for child, child_update in glob_children:
                    inner = normalize_path(outer + path + (child,))
                    if isinstance(child_update, dict):
                        _merge_at(inverse, inner, child_update)
                    else:
                        assoc_path(inverse, inner, child_update)

        elif key in update:
            value = update[key]
            if isinstance(path, dict):
                if '_path' in path:
                    path = path.copy()
                    inner = normalize_path(outer + path.pop('_path'))

                    if not isinstance(value, dict):
                        # A leaf value is routed directly to the store at
                        # ``_path`` instead of to keyed sub-stores.
                        assoc_path(inverse, inner, value)
                        continue

                    for update_key in value.keys():
                        if update_key not in path and '*' not in path:
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
                    _merge_at(inverse, inner, value)
                else:
                    assoc_path(inverse, inner, value)
    return inverse


def _merge_at(tree, path, value):
    """Deep-merge dict ``value`` into ``tree`` at ``path``, in place.

    Unlike ``update_in``, this doesn't copy every dict along ``path``, so
    building an inverse stays linear in the number of ports. Dicts from
    ``value`` are copied as they are inserted instead, so ``tree`` never
    shares a dict with ``value`` and later merges can't write into the
    caller's update.
    """
    node = tree
    for step in path:
        node = node.setdefault(step, {})
    _merge_copy(node, value)


def _merge_copy(dct, merge_dct):
    for key, value in merge_dct.items():
        existing = dct.get(key)
        if isinstance(existing, dict) \
                and isinstance(value, collections.abc.Mapping):
            _merge_copy(existing, value)
        elif isinstance(value, dict):
            dct[key] = _merge_copy({}, value)
        else:
            dct[key] = value
    return dct


def normalize_path(path):
    """Resolve ``..`` elements against the steps before them.

    A ``..`` with nothing left to climb out of is kept, so a path that
    climbs above its start keeps all of its leading ``..`` steps.

    >>> normalize_path(('a', 'b', '..', 'c'))
    ('a', 'c')
    >>> normalize_path(('a', '..', '..', '..', 'x'))
    ('..', '..', 'x')
    """
    progress = []
    for step in path:
        if step == '..' and progress and progress[-1] != '..':
            progress.pop()
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


def test_inverse_topology_leaves_update_untouched():
    # nested and repeated routes merge into the inverse without writing
    # into, or sharing dicts with, the caller's update
    update = {
        'a': {'x': {'y': 1}},
        'b': {'z': 2},
        'c': {'x': {'w': 3}},
        'g': {'0': {'m': {'k': 4}}, '1': {'m': {'k': 5}}}}
    original = copy.deepcopy(update)
    topology = {
        'a': ('s',),
        'b': ('s', 'x'),
        'c': ('s',),
        'g': {'*': {'m': ('t',), 'n': ('t', 'deeper')}}}
    inverse = inverse_topology((), update, topology)
    assert inverse == {
        's': {'x': {'y': 1, 'z': 2, 'w': 3}},
        '0': {'t': {'k': 4}},
        '1': {'t': {'k': 5}}}
    assert update == original

    inverse['s']['x']['new'] = True
    inverse['0']['t']['new'] = True
    assert update == original


def test_inverse_topology_glob_skips_named_keys():
    # a named port next to a top-level glob is not also a glob child
    inverse = inverse_topology(
        (),
        {'special': {'mass': 100}, '1': {'mass': 1}},
        {'*': ('cells',), 'special': ('cells', '0')})
    assert inverse == {'cells': {'0': {'mass': 100}, '1': {'mass': 1}}}

    # same for a glob nested under a '_path' port, with a dict value...
    topology = {
        'agents': {
            '_path': ('cells',),
            '*': {'mass': ('mass',)},
            'special': ('special_store',)}}
    inverse = inverse_topology(
        (), {'agents': {'special': {'mass': 5}, '0': {'mass': 1}}}, topology)
    assert inverse == {
        'cells': {'0': {'mass': 1}, 'special_store': {'mass': 5}}}

    # ...and with a leaf value, which used to raise a TypeError
    inverse = inverse_topology(
        (), {'agents': {'special': 5, '0': {'mass': 1}}}, topology)
    assert inverse == {'cells': {'0': {'mass': 1}, 'special_store': 5}}


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
