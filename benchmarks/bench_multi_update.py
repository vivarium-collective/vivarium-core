"""Benchmark how process topologies affect engine step time.

Uses only the public Engine API, so the same script runs against any
checkout of vivarium-core (e.g. master vs. a branch). It imports vivarium
from the checkout given by --repo, not from wherever it is installed.

Run against one checkout and save the results:

    python benchmarks/bench_multi_update.py --repo . --save branch.json

Compare two saved runs:

    python benchmarks/bench_multi_update.py --compare master.json branch.json
"""

import argparse
import gc
import json
import logging
import os
import statistics
import subprocess
import sys
import time


# (label, description) for each topology shape; see make_topology
SHAPES = [
    ('tuples', 'every port a tuple path, no overlap (fast path)'),
    ('bare_path', "bulk port is a bare {'_path': ...}"),
    ('path_routes', "bulk port is {'_path': ..., 'm0': ...}"),
    ('path_overlap', "path_routes, plus ports p0 and p1 both on store s0"),
    ('overlap', 'tuples, plus ports p0 and p1 both on store s0'),
    ('overlap_leaf', "tuples, plus p0 and p1 as {'v': ('s0', 'v')} "
                     '(leaf overlap: issue #209)'),
    ('glob', "tuples, plus a top-level glob '*': ('cells',) over n children"),
]


def make_process_class(Process):
    class Many(Process):
        """``n`` scalar ports plus one port holding ``n`` variables."""
        name = 'many'
        defaults = {'n': 10, 'glob': False}

        def ports_schema(self):
            n = self.parameters['n']
            schema = {
                'bulk': {f'm{i}': {'_default': 0} for i in range(n)},
                **{f'p{i}': {'v': {'_default': 0}} for i in range(n)}}
            if self.parameters['glob']:
                schema['*'] = {'v': {'_default': 0}}
            return schema

        def next_update(self, timestep, states):
            n = self.parameters['n']
            update = {
                'bulk': {f'm{i}': 1 for i in range(n)},
                **{f'p{i}': {'v': 1} for i in range(n)}}
            if self.parameters['glob']:
                update.update({f'c{i}': {'v': 1} for i in range(n)})
            return update

    return Many


def make_topology(shape, n):
    ports = {f'p{i}': (f's{i}',) for i in range(n)}
    if shape in ('overlap', 'path_overlap'):
        ports['p1'] = ('s0',)
    elif shape == 'overlap_leaf':
        ports['p0'] = ports['p1'] = {'v': ('s0', 'v')}
    bulk = {
        'tuples': ('bulk',),
        'overlap': ('bulk',),
        'overlap_leaf': ('bulk',),
        'bare_path': {'_path': ('bulk',)},
        'path_routes': {'_path': ('bulk',), 'm0': ('m0',)},
        'path_overlap': {'_path': ('bulk',), 'm0': ('m0',)},
        'glob': ('bulk',),
    }[shape]
    topology = {'bulk': bulk, **ports}
    if shape == 'glob':
        topology['*'] = ('cells',)
    return topology


def expected_s0(shape, steps):
    # p0 and p1 both add 1 to s0/v each step when they overlap
    overlapping = ('overlap', 'path_overlap', 'overlap_leaf')
    return 2 * steps if shape in overlapping else steps


def bench_case(Engine, Many, shape, n, steps, repeats, gc_enabled):
    glob = shape == 'glob'
    exp = Engine(
        processes={'proc': Many({'n': n, 'glob': glob})},
        topology={'proc': make_topology(shape, n)},
        initial_state=(
            {'cells': {f'c{i}': {'v': 0} for i in range(n)}} if glob else None),
        emitter='null',
        display_info=False,
        progress_bar=False)
    exp.update(1)  # warm up (builds caches) outside the timed runs
    times = []
    try:
        if not gc_enabled:
            gc.disable()
        for _ in range(repeats):
            start = time.perf_counter()
            exp.update(steps)
            times.append((time.perf_counter() - start) / steps * 1e3)
    finally:
        gc.enable()

    total_steps = 1 + repeats * steps
    s0 = exp.state.get_path(('s0',)).get_value()['v']
    correct = s0 == expected_s0(shape, total_steps)
    if glob:
        correct = correct and exp.state.get_path(
            ('cells', 'c0', 'v')).get_value() == total_steps
    return {
        'median_ms': statistics.median(times),
        'min_ms': min(times),
        # False means a contribution was silently dropped
        'correct': correct,
    }


def git_describe(repo):
    try:
        return subprocess.run(
            ['git', '-C', repo, 'describe', '--always', '--dirty'],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def run(args):
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    logging.disable(logging.WARNING)
    import vivarium  # pylint: disable=import-outside-toplevel
    from vivarium.core.engine import Engine  # pylint: disable=import-outside-toplevel
    from vivarium.core.process import Process  # pylint: disable=import-outside-toplevel

    vivarium_dir = os.path.dirname(os.path.dirname(vivarium.__file__))
    if os.path.realpath(vivarium_dir) != os.path.realpath(repo):
        sys.exit(f'imported vivarium from {vivarium_dir}, not {repo}')

    Many = make_process_class(Process)
    label = args.label or git_describe(repo)
    print(f'vivarium from {vivarium_dir} ({label})')
    print(f'gc {"on" if args.gc else "off"}, median of {args.repeats} runs\n')
    print(f'{"n":>6} {"shape":<14} {"median ms/step":>15} {"min":>9}  correct')

    results = []
    for n in args.sizes:
        # aim for roughly the same wall time per size
        steps = max(1, args.step_budget // n)
        for shape, _ in SHAPES:
            result = bench_case(
                Engine, Many, shape, n, steps, args.repeats, args.gc)
            result.update({'n': n, 'shape': shape})
            results.append(result)
            print(f'{n:>6} {shape:<14} {result["median_ms"]:>15.3f} '
                  f'{result["min_ms"]:>9.3f}  {result["correct"]}')

    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump({'label': label, 'gc': args.gc,
                       'results': results}, f, indent=2)
        print(f'\nsaved to {args.save}')


def compare(base_path, new_path):
    with open(base_path, encoding='utf-8') as f:
        base = json.load(f)
    with open(new_path, encoding='utf-8') as f:
        new = json.load(f)
    base_by_key = {(r['n'], r['shape']): r for r in base['results']}

    print(f'base: {base["label"]}   new: {new["label"]}\n')
    print(f'{"n":>6} {"shape":<14} {"base ms":>9} {"new ms":>9} '
          f'{"speedup":>8}  correct (base -> new)')
    for r in new['results']:
        b = base_by_key.get((r['n'], r['shape']))
        if b is None:
            continue
        speedup = b['median_ms'] / r['median_ms']
        print(f'{r["n"]:>6} {r["shape"]:<14} {b["median_ms"]:>9.3f} '
              f'{r["median_ms"]:>9.3f} {speedup:>7.2f}x  '
              f'{b["correct"]} -> {r["correct"]}')
    print('\nshapes:')
    for shape, description in SHAPES:
        print(f'  {shape:<14} {description}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '--repo', default='.',
        help='checkout of vivarium-core to benchmark (default: .)')
    parser.add_argument('--save', help='write results to this JSON file')
    parser.add_argument('--label', help='name for this run (default: git describe)')
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=[10, 100, 500, 2000],
        help='port counts to benchmark (default: 10 100 500 2000)')
    parser.add_argument(
        '--repeats', type=int, default=5,
        help='timed runs per case; the median is reported (default: 5)')
    parser.add_argument(
        '--step-budget', type=int, default=2000,
        help='steps per run is step_budget / n (default: 2000)')
    parser.add_argument(
        '--no-gc', dest='gc', action='store_false',
        help='disable garbage collection during timed runs')
    parser.add_argument(
        '--compare', nargs=2, metavar=('BASE', 'NEW'),
        help='compare two saved JSON results instead of running')
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == '__main__':
    main()
