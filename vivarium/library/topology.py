import copy

from vivarium.library.dict_utils import deep_merge, deep_merge_multi_update


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
                                current, child_update))
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

                    for update_key in update[key].keys():
                        if update_key not in path:
                            path[update_key] = (update_key,)

                deep_merge(
                    node,
                    inverse_topology(
                        tuple(),
                        value,
                        path))

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
    return progress
