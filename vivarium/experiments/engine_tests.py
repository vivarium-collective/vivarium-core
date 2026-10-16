import random
import logging as log
from typing import Optional, Union, Dict, Any, cast, List

from vivarium.composites.toys import (
    PoQo, Sine, ToyDivider, ToyTransport, ToyEnvironment, ToyProcess,
    Proton, Electron, MoveProcess)
from vivarium.core.composer import Composer, Composite
from vivarium.core.engine import (
    Engine, pf, pp, _StepGraph, invert_topology, topology_can_overlap,
    topology_split_plan)
from vivarium.core.process import Process, Step, Deriver
from vivarium.core.store import Store, hierarchy_depth
from vivarium.core.types import (
    Schema, State, Update, Topology, Steps, Processes)
from vivarium.library.units import units
from vivarium.library.wrappers import make_logging_process
from vivarium.core.control import run_library_cli


def _make_proton(
        parallel: bool = False
) -> Dict[str, Any]:
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


def test_recursive_store() -> None:
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


def test_topology_ports() -> None:
    proton = _make_proton()

    experiment = Engine(**proton)

    log.debug(pf(experiment.state.get_config(True)))

    experiment.update(10.0)

    log.debug(pf(experiment.state.get_config(True)))
    log.debug(pf(experiment.state.divide_value()))


def test_timescales() -> None:
    class Slow(Process):
        name = 'slow'
        defaults = {'timestep': 3.0}

        def __init__(self, config: Optional[dict] = None) -> None:
            super().__init__(config)

        def ports_schema(self) -> Schema:
            return {
                'state': {
                    'base': {
                        '_default': 1.0}}}

        def next_update(
                self,
                timestep: Union[float, int],
                states: State) -> Update:
            base = states['state']['base']
            next_base = timestep * base * 0.1

            return {
                'state': {'base': next_base}}

    class Fast(Process):
        name = 'fast'
        defaults = {'timestep': 0.3}

        def __init__(self, config: Optional[dict] = None) -> None:
            super().__init__(config)

        def ports_schema(self) -> Schema:
            return {
                'state': {
                    'base': {
                        '_default': 1.0},
                    'motion': {
                        '_default': 0.0}}}

        def next_update(
                self,
                timestep: Union[float, int],
                states: State) -> Update:
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

    topology: Topology = {
        'slow': {'state': ('state',)},
        'fast': {'state': ('state',)}}

    emitter = {'type': 'null'}
    experiment = Engine(
        processes=processes,
        topology=topology,
        emitter=emitter,
        initial_state=states)

    experiment.update(10.0)


def test_2_store_1_port() -> None:
    """
    Split one port of a processes into two stores
    """

    class OnePort(Process):
        name = 'one_port'

        def ports_schema(self) -> Schema:
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

        def next_update(
                self,
                timestep: Union[float, int],
                states: State) -> Update:
            return {
                'A': {
                    'a': 1,
                    'b': 2}}

    class SplitPort(Composer):
        """splits OnePort's ports into two stores"""
        name = 'split_port_composer'

        def generate_processes(
                self, config: Optional[dict]) -> Dict[str, Any]:
            return {
                'one_port': OnePort({})}

        def generate_topology(self, config: Optional[dict]) -> Topology:
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
    exp = Engine(**{
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'external': {'a': [0, 2, 4]},
        'internal': {'a': [0, 1, 2]},
        'time': [0.0, 1.0, 2.0]}
    assert output == expected_output


class MultiPort(Process):
    name = 'multi_port'

    def ports_schema(self) -> Schema:
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

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            'A': {'a': 1},
            'B': {'a': 1},
            'C': {'a': 1}}


class MergePort(Composer):
    """combines both of MultiPort's ports into one store"""
    name = 'multi_port_composer'

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {
            'multi_port': MultiPort({})}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'multi_port': {
                'A': ('aaa',),
                'B': ('aaa',),
                'C': ('aaa',)}}


def test_multi_port_merge() -> None:
    # run experiment
    merge_port = MergePort({})
    network = merge_port.generate()
    exp = Engine(**{
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'aaa': {'a': [0, 3, 6]},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class DictTopologyMultiPort(Process):
    """Two ports that point to the same store via ``_path`` dict topologies
    and update the same variable."""
    name = 'dict_topology_multi_port'

    def ports_schema(self) -> Schema:
        return {
            'A': {
                'a': {
                    '_default': 0,
                    '_emit': True}},
            'B': {
                'a': {
                    '_default': 0,
                    '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            'A': {'a': 1},
            'B': {'a': 1}}


class DictTopologyComposer(Composer):
    """Merges both of DictTopologyMultiPort's ports into one store using
    ``_path`` dict topologies (see issue #209)."""
    name = 'dict_topology_composer'

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {
            'multi_port': DictTopologyMultiPort({})}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'multi_port': {
                'A': {'_path': ('shared',)},
                'B': {'_path': ('shared',)}
            }
        }

def test_dict_topology_multi_port() -> None:
    """Overlapping ports with dict (``_path``) topologies must aggregate
    their updates instead of silently dropping one of them."""
    composer = DictTopologyComposer({})
    network = composer.generate()
    exp = Engine(
        processes=network['processes'],
        topology=network['topology'])

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'shared': {'a': [0, 2, 4]},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class InitialStateMultiPort(Process):
    """Two ports to the same store that also seed it via ``initial_state``
    (see issue #209)."""
    name = 'initial_state_multi_port'

    def ports_schema(self) -> Schema:
        return {
            'A': {
                'a': {
                    '_default': 0,
                    '_emit': True}},
            'B': {
                'a': {
                    '_default': 0,
                    '_emit': True}}}

    def initial_state(self, config: Optional[dict] = None) -> State:
        return {
            'A': {'a': 5},
            'B': {'a': 5}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            'A': {'a': 1},
            'B': {'a': 1}}


class InitialStateComposer(Composer):
    """Merges both of InitialStateMultiPort's ports into one store."""
    name = 'initial_state_composer'

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {
            'multi_port': InitialStateMultiPort({})}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'multi_port': {
                'A': ('shared',),
                'B': ('shared',)}}


def test_multi_port_initial_state() -> None:
    """A process with overlapping ports must be able to provide an
    ``initial_state`` without forming multi-update lists (see issue #209
    and PR #205)."""
    composer = InitialStateComposer({})
    network = composer.generate()
    exp = Engine(
        processes=network['processes'],
        topology=network['topology'],
        initial_state=composer.initial_state())

    exp.update(2)
    output = exp.emitter.get_timeseries()
    # The two ports deep-merge to a single starting value of 5, then each
    # step adds 1 per port.
    expected_output = {
        'shared': {'a': [5, 7, 9]},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class InitialStateDirectLeafComposer(Composer):
    """Merges both of InitialStateMultiPort's ports into one store."""
    name = 'initial_state_direct_leaf_composer'

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {
            'multi_port': InitialStateMultiPort({})}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'multi_port': {
                'A': {'a': ('shared',)},
                'B': {'a': ('shared',)}}}


def test_multi_port_initial_state_direct_leaf() -> None:
    """A process with overlapping ports must be able to provide an
    ``initial_state`` without forming multi-update lists (see issue #209
    and PR #205) even for direct leaf ports."""
    composer = InitialStateDirectLeafComposer({})
    network = composer.generate()
    exp = Engine(
        processes=network['processes'],
        topology=network['topology'],
        initial_state=composer.initial_state())

    exp.update(2)
    output = exp.emitter.get_timeseries()
    # The two ports deep-merge to a single starting value of 5, then each
    # step adds 1 per port.
    expected_output = {
        'shared': [5, 7, 9],
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class DictValueMultiPort(Process):
    """Two ports to the same dict-valued store, each updating different
    keys (see issue #209)."""
    name = 'dict_value_multi_port'

    def ports_schema(self) -> Schema:
        return {
            'A': {
                'a': {
                    '_default': {},
                    '_updater': 'dict_value',
                    '_emit': True}},
            'B': {
                'a': {
                    '_default': {},
                    '_updater': 'dict_value',
                    '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            'A': {'a': {'_add': [{'key': 'x', 'state': 1}]}},
            'B': {'a': {'_add': [{'key': 'y', 'state': 2}]}}}


class DictValueComposer(Composer):
    """Merges both of DictValueMultiPort's ports into one dict-valued
    store."""
    name = 'dict_value_composer'

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {
            'multi_port': DictValueMultiPort({})}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'multi_port': {
                'A': ('shared',),
                'B': ('shared',)}}


def test_multi_port_dict_values() -> None:
    """Overlapping ports whose shared store holds ``dict`` values must
    merge both updates rather than break the multi-update aggregation
    (see issue #209)."""
    composer = DictValueComposer({})
    network = composer.generate()
    exp = Engine(
        processes=network['processes'],
        topology=network['topology'])

    exp.update(1)
    data = exp.emitter.get_data()
    # Both ports' updates land in the same dict store.
    assert data[1.0]['shared']['a'] == {'x': 1, 'y': 2}


class MixedTopologyMultiPort(Process):
    """Two ports to the same store, one wired with a tuple topology and
    the other with a ``_path`` dict topology (see issue #209)."""
    name = 'mixed_topology_multi_port'

    def ports_schema(self) -> Schema:
        return {
            'A': {
                'a': {
                    '_default': 0,
                    '_emit': True}},
            'B': {
                'a': {
                    '_default': 0,
                    '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            'A': {'a': 1},
            'B': {'a': 1}}


class MixedTopologyComposer(Composer):
    """Merges MixedTopologyMultiPort's ports into one store, mixing tuple
    and dict topologies."""
    name = 'mixed_topology_composer'

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {
            'multi_port': MixedTopologyMultiPort({})}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'multi_port': {
                'A': ('shared',),
                'B': {'_path': ('shared',)}}}


def test_mixed_topology_multi_port() -> None:
    """Overlapping ports must aggregate correctly even when one port uses
    a tuple topology and another uses a dict (``_path``) topology (see
    issue #209)."""
    composer = MixedTopologyComposer({})
    network = composer.generate()
    exp = Engine(
        processes=network['processes'],
        topology=network['topology'])

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'shared': {'a': [0, 2, 4]},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class DirectLeafMultiPort(Process):
    """Two ports that each point directly at a single leaf store instead
    of a subtree (see issue #209)."""
    name = 'direct_leaf_multi_port'

    def ports_schema(self) -> Schema:
        return {
            'A': {
                '_default': 0,
                '_emit': True},
            'B': {
                '_default': 0,
                '_emit': True}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            'A': 1,
            'B': 1}


class DirectLeafComposer(Composer):
    """Merges both of DirectLeafMultiPort's leaf ports into one store."""
    name = 'direct_leaf_composer'

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {
            'multi_port': DirectLeafMultiPort({})}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'multi_port': {
                'A': ('shared',),
                'B': ('shared',)}}


def test_direct_leaf_multi_port() -> None:
    """Overlapping ports that point directly at a single leaf store must
    aggregate their updates (see issue #209)."""
    composer = DirectLeafComposer({})
    network = composer.generate()
    exp = Engine(
        processes=network['processes'],
        topology=network['topology'])

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'shared': [0, 2, 4],
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class PartialUpdateMultiPort(Process):
    """Three ports to the same store, but only some are updated each step
    (see issue #209)."""
    name = 'partial_update_multi_port'

    def ports_schema(self) -> Schema:
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

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        # Port C is intentionally omitted from the update.
        return {
            'A': {'a': 1},
            'B': {'a': 1}}


class PartialUpdateComposer(Composer):
    """Merges all of PartialUpdateMultiPort's ports into one store."""
    name = 'partial_update_composer'

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {
            'multi_port': PartialUpdateMultiPort({})}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'multi_port': {
                'A': ('shared',),
                'B': ('shared',),
                'C': ('shared',)}}


def test_partial_update_multi_port() -> None:
    """Overlapping ports that are omitted from an update must be skipped
    without dropping the ports that are present (see issue #209)."""
    composer = PartialUpdateComposer({})
    network = composer.generate()
    exp = Engine(
        processes=network['processes'],
        topology=network['topology'])

    exp.update(2)
    output = exp.emitter.get_timeseries()
    # Only ports A and B contribute, so the store grows by 2 each step.
    expected_output = {
        'shared': {'a': [0, 2, 4]},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class SiblingVarMultiPort(Process):
    """Two ports to the same store that write different sibling variables
    (see issue #209)."""
    name = 'sibling_var_multi_port'

    def ports_schema(self) -> Schema:
        return {
            'A': {
                'x': {
                    '_default': 0,
                    '_emit': True}},
            'B': {
                'y': {
                    '_default': 0,
                    '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            'A': {'x': 1},
            'B': {'y': 2}}


class SiblingVarComposer(Composer):
    """Merges both of SiblingVarMultiPort's ports into one store."""
    name = 'sibling_var_composer'

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {
            'multi_port': SiblingVarMultiPort({})}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'multi_port': {
                'A': ('shared',),
                'B': ('shared',)}}


def test_sibling_var_multi_port() -> None:
    """Splitting updates per port must not clobber sibling variables that
    live in the same shared store (see issue #209)."""
    composer = SiblingVarComposer({})
    network = composer.generate()
    exp = Engine(
        processes=network['processes'],
        topology=network['topology'])

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'shared': {
            'x': [0, 1, 2],
            'y': [0, 2, 4]},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class NestedSubtreeMultiPort(Process):
    """Two ports to the same store that write different nested sub-paths
    (see issue #209)."""
    name = 'nested_subtree_multi_port'

    def ports_schema(self) -> Schema:
        return {
            'A': {
                'sub': {
                    'x': {
                        '_default': 0,
                        '_emit': True}}},
            'B': {
                'sub': {
                    'y': {
                        '_default': 0,
                        '_emit': True}}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            'A': {'sub': {'x': 1}},
            'B': {'sub': {'y': 2}}}


class NestedSubtreeComposer(Composer):
    """Merges both of NestedSubtreeMultiPort's ports into one store."""
    name = 'nested_subtree_composer'

    def generate_processes(
            self, config: Optional[dict]) -> Dict[str, Any]:
        return {
            'multi_port': NestedSubtreeMultiPort({})}

    def generate_topology(self, config: Optional[dict]) -> Topology:
        return {
            'multi_port': {
                'A': ('shared',),
                'B': ('shared',)}}


def test_nested_subtree_multi_port() -> None:
    """Overlapping ports that write different nested sub-paths of the same
    store must deep-merge instead of overwriting (see issue #209)."""
    composer = NestedSubtreeComposer({})
    network = composer.generate()
    exp = Engine(
        processes=network['processes'],
        topology=network['topology'])

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        'shared': {
            'sub': {
                'x': [0, 1, 2],
                'y': [0, 2, 4]}},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class GlobRootProcess(Process):
    """A process whose entire topology is a glob (``*``) port, exercising
    the glob fallback in ``invert_topology`` (see issue #209)."""
    name = 'glob_root_process'

    def ports_schema(self) -> Schema:
        return {
            '*': {
                'mass': {
                    '_default': 0,
                    '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            agent_id: {'mass': 1}
            for agent_id in states}


def test_glob_root_multi_port() -> None:
    """A top-level glob port must fall back to single-update handling and
    still distribute its update across every child (see issue #209)."""
    process = GlobRootProcess({})
    exp = Engine(
        processes={'glob': process},
        topology={'glob': {'*': ('agents',)}},
        initial_state={
            'agents': {
                '0': {'mass': 0},
                '1': {'mass': 0}}})

    exp.update(2)
    data = exp.emitter.get_data()
    assert data[2.0]['agents']['0']['mass'] == 2
    assert data[2.0]['agents']['1']['mass'] == 2


class GlobPlusDirectProcess(Process):
    """A process with a top-level glob (``*``) port and a named port that
    points directly at one of the glob's children, updating it too (see
    issue #209)."""
    name = 'glob_plus_direct_process'

    def ports_schema(self) -> Schema:
        return {
            '*': {
                'mass': {
                    '_default': 0,
                    '_emit': True}},
            'special': {
                'mass': {
                    '_default': 0,
                    '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        update: Update = {'special': {'mass': 100}}
        for agent_id in states:
            if agent_id == 'special':
                continue
            update[agent_id] = {'mass': 1}
        return update


def test_glob_plus_direct_port() -> None:
    """A named port must be split out from a glob port so that its update
    is not mistaken for a glob child nor aggregated into a fragile
    multi-update list (see issue #209)."""
    process = GlobPlusDirectProcess({})
    exp = Engine(
        processes={'proc': process},
        topology={
            'proc': {
                '*': ('cells',),
                'special': ('cells', '0')}},
        initial_state={
            'cells': {
                '0': {'mass': 0},
                '1': {'mass': 0}}})

    exp.update(1)
    data = exp.emitter.get_data()
    # cell '0' gets both the glob update (+1) and the direct update (+100).
    assert data[1.0]['cells']['0']['mass'] == 101
    assert data[1.0]['cells']['1']['mass'] == 1
    # The named port must not leak in as a spurious glob child.
    assert 'special' not in data[1.0]['cells']


class IntraPortOverlapProcess(Process):
    """A single port whose two variables are wired to the same store, so
    the overlap happens within one port rather than across ports (see
    issue #209)."""
    name = 'intra_port_overlap_process'

    def __init__(self, parameters: Optional[dict] = None) -> None:
        super().__init__(parameters)
        self.update = self.parameters['update']

    def ports_schema(self) -> Schema:
        return {
            'port': {
                'x': {'_default': 0, '_emit': True},
                'y': {'_default': 0, '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {'port': self.update}


class IntraPortDictProcess(Process):
    """A single port whose two branch variables are wired to the same
    store and both update its ``m`` leaf (see issue #209)."""
    name = 'intra_port_dict_process'

    def ports_schema(self) -> Schema:
        return {
            'port': {
                'x': {'m': {'_default': 0, '_emit': True}},
                'y': {'m': {'_default': 0, '_emit': True}}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {'port': {'x': {'m': 1}, 'y': {'m': 2}}}


def test_intra_port_scalar_overlap() -> None:
    """Two scalar variables of one port wired to the same leaf must both
    be applied instead of one overwriting the other (see issue #209)."""
    process = IntraPortOverlapProcess({'update': {'x': 1, 'y': 2}})
    exp = Engine(
        processes={'proc': process},
        topology={'proc': {'port': {'x': ('shared',), 'y': ('shared',)}}},
        initial_state={'shared': 0})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    # Each step accumulates x (+1) and y (+2) onto the same store.
    expected_output = {
        'shared': [0, 3, 6],
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


def test_intra_port_dict_overlap() -> None:
    """Two dict-valued variables of one port wired to the same store must
    both be applied through the store's updater rather than aggregated
    into a multi-update list (see issue #209)."""
    process = IntraPortDictProcess({})
    exp = Engine(
        processes={'proc': process},
        topology={'proc': {'port': {'x': ('shared',), 'y': ('shared',)}}},
        initial_state={'shared': {'m': 0}})

    exp.update(1)
    data = exp.emitter.get_data()
    # Both contributions reach the same leaf and accumulate: 1 + 2 = 3.
    assert data[1.0]['shared']['m'] == 3


class PathRouteOverlapProcess(Process):
    """A single port whose ``_path`` topology routes two explicit variables
    to the same leaf under the path (see issue #209)."""
    name = 'path_route_overlap_process'

    def ports_schema(self) -> Schema:
        return {
            'port': {
                'x': {'_default': 0, '_emit': True},
                'y': {'_default': 0, '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {'port': {'x': 1, 'y': 2}}


def test_path_route_overlap() -> None:
    """Two explicit routes of a ``_path`` topology that land on the same
    leaf must both be applied instead of one overwriting the other (see
    issue #209)."""
    process = PathRouteOverlapProcess({})
    exp = Engine(
        processes={'proc': process},
        topology={
            'proc': {
                'port': {
                    '_path': ('bulk',),
                    'x': ('m',),
                    'y': ('m',)}}},
        initial_state={'bulk': {'m': 0}})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    # Each step accumulates x (+1) and y (+2) onto bulk/m.
    expected_output = {
        'bulk': {'m': [0, 3, 6]},
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class PathDirectiveProcess(Process):
    """A port whose ``_path`` topology carries an atomic store directive
    ({'_updater', '_value'}); it must not be split (see issue #209)."""
    name = 'path_directive_process'

    def ports_schema(self) -> Schema:
        return {
            'trigger': {
                '_default': [],
                '_updater': 'set',
                '_emit': True}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {'trigger': {'_updater': 'set', '_value': [1, 2, 3]}}


def test_path_directive_not_split() -> None:
    """An atomic ``{'_updater', '_value'}`` directive routed through a
    ``_path`` topology must stay intact rather than being split into
    separate updates (see issue #209)."""
    process = PathDirectiveProcess({})
    exp = Engine(
        processes={'proc': process},
        topology={'proc': {'trigger': {'_path': ('shared',)}}},
        initial_state={'shared': []})

    exp.update(1)
    data = exp.emitter.get_data()
    assert data[1.0]['shared'] == [1, 2, 3]


class PathRewireProcess(Process):
    """A port with a ``_path`` base plus explicit routes: an unlisted
    variable defaults under the base, a listed route is resolved relative
    to the base, and a listed route may use ``..`` to reach a separate
    store outside the base."""
    name = 'path_rewire_process'

    def ports_schema(self) -> Schema:
        return {
            'port': {
                'base_var': {'_default': 0, '_emit': True},
                'near_var': {'_default': 0, '_emit': True},
                'far_var': {'_default': 0, '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {'port': {'base_var': 1, 'near_var': 2, 'far_var': 3}}


def test_path_base_rooted_routes() -> None:
    """Explicit ``_path`` routes are resolved relative to the ``_path``
    base; a route reaches a separate store by using ``..`` to climb out of
    the base (as ``PoQo`` does). The split must preserve these semantics."""
    process = PathRewireProcess({})
    exp = Engine(
        processes={'proc': process},
        topology={
            'proc': {
                'port': {
                    '_path': ('base',),
                    # relative to the base -> base/near
                    'near_var': ('near',),
                    # '..' climbs out of the base -> far
                    'far_var': ('..', 'far')}}},
        initial_state={'base': {'base_var': 0, 'near': 0}, 'far': 0})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    expected_output = {
        # unlisted variable defaults under the base
        'base': {
            'base_var': [0, 1, 2],
            'near': [0, 2, 4]},
        'far': [0, 3, 6],
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class OneOffUpdaterProcess(Process):
    """Returns a one-off ``_updater``/``_value`` override in its update,
    replacing the store's default updater for that update only."""
    name = 'one_off_updater_process'

    def ports_schema(self) -> Schema:
        return {
            'port': {
                'x': {
                    '_default': 0,
                    '_updater': 'accumulate',
                    '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        # Override the default 'accumulate' with a one-off 'set' to 100.
        return {'port': {'x': {'_updater': 'set', '_value': 100}}}


def test_one_off_updater_override() -> None:
    """An update may pair ``_updater`` with ``_value`` to override the
    store's default updater for that update; the two keys must reach the
    store together through the split."""
    process = OneOffUpdaterProcess({})
    exp = Engine(
        processes={'proc': process},
        topology={'proc': {'port': {'x': ('shared',)}}},
        initial_state={'shared': 0})

    exp.update(2)
    output = exp.emitter.get_timeseries()
    # 'set' to 100 each step instead of accumulating.
    expected_output = {
        'shared': [0, 100, 100],
        'time': [0.0, 1.0, 2.0]}

    assert output == expected_output


class RewireOverlapProcess(Process):
    """Two ports that can be rewired onto the same store to create an
    overlap after the engine has started (see issue #209)."""
    name = 'rewire_overlap_process'

    def ports_schema(self) -> Schema:
        return {
            'port1': {'x': {'_default': 0, '_emit': True}},
            'port2': {'x': {'_default': 0, '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {'port1': {'x': 1}, 'port2': {'x': 10}}


def test_invert_cache_invalidated_on_rewire() -> None:
    """Rewiring a port to overlap another must invalidate the cached
    "can overlap" decision, so the update is split and no contribution is
    lost (see issue #209)."""
    process = RewireOverlapProcess({})
    exp = Engine(
        processes={'proc': process},
        topology={'proc': {'port1': ('s1',), 'port2': ('s2',)}},
        initial_state={'s1': {'x': 0}, 's2': {'x': 0}})

    # ports go to distinct stores, so the fast non-split path is cached
    exp.update(1)
    proc_store = exp.state.get_path(('proc',))
    assert proc_store.invert_can_overlap is False
    assert exp.state.get_path(('s1',)).get_value() == {'x': 1}

    # rewire port2 onto s1, creating an overlap
    proc_store.connect('port2', exp.state.get_path(('s1',)))

    exp.update(1)
    # both ports now hit s1 and must both apply: +1 and +10
    assert proc_store.invert_can_overlap is True  # recomputed as overlapping
    assert exp.state.get_path(('s1',)).get_value() == {'x': 12}
    # s2 is no longer wired, so it keeps its earlier value
    assert exp.state.get_path(('s2',)).get_value() == {'x': 10}


def test_topology_edits_invalidate_cache() -> None:
    """Any change to a process store's topology -- reassignment, or an
    in-place edit at any depth, including through the dict originally
    passed to the Engine -- must make the engine recompute the cached
    overlap decision, so a manual rewire can't leave a stale fast-path
    verdict behind (see issue #209)."""

    def rewire_port2_onto_s1(topology: dict, store: Store) -> None:
        topology['proc']['port2'] = ('s1',)

    def rewire_via_store(topology: dict, store: Store) -> None:
        store.topology['port2'] = ('s1',)

    def rewire_nested_in_place(topology: dict, store: Store) -> None:
        store.topology['port2']['x'] = ('s1', 'x')

    def reassign(topology: dict, store: Store) -> None:
        store.topology = {'port1': ('s1',), 'port2': ('s1',)}

    cases: List[Any] = [
        ('original dict', {'port2': ('s2',)}, rewire_port2_onto_s1),
        ('store, top level', {'port2': ('s2',)}, rewire_via_store),
        ('store, nested', {'port2': {'x': ('s2', 'x')}},
         rewire_nested_in_place),
        ('reassignment', {'port2': ('s2',)}, reassign),
    ]
    for label, port2_topology, edit in cases:
        topology: Dict[str, Any] = {
            'proc': {'port1': ('s1',), **port2_topology}}
        exp = Engine(
            processes={'proc': RewireOverlapProcess({})},
            topology=topology,
            initial_state={'s1': {'x': 0}, 's2': {'x': 0}})
        proc_store = exp.state.get_path(('proc',))

        # distinct stores, so the fast path is cached; an unchanged
        # topology keeps the same snapshot instead of recomputing
        exp.update(1)
        assert proc_store.invert_can_overlap is False, label
        snapshot = proc_store.invert_topology_snapshot
        exp.update(1)
        assert proc_store.invert_topology_snapshot is snapshot, label
        assert exp.state.get_path(('s1',)).get_value() == {'x': 2}, label

        # the edit creates an overlap on s1, so both ports must apply
        # (+1 and +10); a stale verdict would drop port1's +1
        edit(topology, proc_store)
        exp.update(1)
        assert proc_store.invert_can_overlap is True, label
        assert exp.state.get_path(('s1',)).get_value() == {'x': 13}, label


class SameLeafSetProcess(Process):
    """Two ports wired to the same leaf, both using the overwriting 'set'
    updater instead of a commutative one like 'accumulate'."""
    name = 'same_leaf_set_process'

    def ports_schema(self) -> Schema:
        return {
            'A': {'x': {'_default': None, '_updater': 'set', '_emit': True}},
            'B': {'x': {'_default': None, '_updater': 'set', '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {'A': {'x': 'from_A'}, 'B': {'x': 'from_B'}}


def test_same_leaf_overlap_ordering() -> None:
    """When two ports resolve to the exact same leaf and the store's
    updater overwrites rather than commutes (e.g. 'set'), the split
    updates are applied in the order their ports appear in the topology
    dict, so the last-listed port wins. This pins down the ordering
    contract left implicit by the #209 fix: it is a function of topology
    declaration order, not of anything about the update's content."""
    process = SameLeafSetProcess({})

    # 'B' listed after 'A' in the topology -> 'B' is applied last and wins.
    exp = Engine(
        processes={'proc': process},
        topology={'proc': {'A': ('shared',), 'B': ('shared',)}},
        initial_state={'shared': {'x': None}})
    exp.update(1)
    assert exp.state.get_path(('shared', 'x')).get_value() == 'from_B'

    # Swapping the topology's declaration order swaps the winner, even
    # though the process's update content is unchanged.
    exp2 = Engine(
        processes={'proc': process},
        topology={'proc': {'B': ('shared',), 'A': ('shared',)}},
        initial_state={'shared': {'x': None}})
    exp2.update(1)
    assert exp2.state.get_path(('shared', 'x')).get_value() == 'from_A'


class OverlappingDeleteProcess(Process):
    """Two ports wired to the same glob store, each independently
    deleting and adding disjoint keys (see issue #209). Without the
    per-port split, the two ports' ``_delete``/``_add`` directives would
    be deep-merged in a single inversion pass, and since they are plain
    lists rather than dicts, the second port's lists would simply
    overwrite the first's instead of both taking effect."""
    name = 'overlapping_delete_process'

    def ports_schema(self) -> Schema:
        return {
            'A': {'*': {'_default': 0, '_emit': True}},
            'B': {'*': {'_default': 0, '_emit': True}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            'A': {'_delete': ['a1'], '_add': [{'key': 'a2', 'state': 1}]},
            'B': {'_delete': ['b1'], '_add': [{'key': 'b2', 'state': 1}]}}


def test_overlapping_delete_not_duplicated_or_dropped() -> None:
    """Each port's ``_delete``/``_add`` directive is contained entirely
    within that port's own split update, so when two overlapping ports
    both delete/add keys in the same store, each key is deleted/added
    exactly once -- neither duplicated across the split pieces nor
    dropped by a merge, as both would silently overwrite in a single
    combined inversion pass (see issue #209)."""
    process = OverlappingDeleteProcess({})
    exp = Engine(
        processes={'proc': process},
        topology={'proc': {'A': ('pool',), 'B': ('pool',)}},
        initial_state={'pool': {'a1': 1, 'b1': 1}})

    exp.update(1)
    assert exp.state.get_path(('pool',)).get_value() == {
        'a2': 1, 'b2': 1}


def test_topology_can_overlap() -> None:
    """The static overlap check flags equal or nested targets wherever
    they sit in the topology, resolves a bare ``_path`` to its base, and
    gives up (assumes overlap) on anything it can't resolve."""
    assert not topology_can_overlap({'a': ('x',), 'b': ('y',)})
    # one target is a path prefix of the other, whatever the input order
    assert topology_can_overlap(
        {'a': ('x', 'y'), 'c': ('w',), 'b': ('x',)})
    assert topology_can_overlap({'a': ('x',), 'b': ('x',)})
    # a string prefix is not a path prefix
    assert not topology_can_overlap({'a': ('x',), 'b': ('xy',)})
    # '..' is resolved before comparing
    assert topology_can_overlap({'a': ('y', '..', 'x'), 'b': ('x',)})
    # mixed key types don't break sorting
    assert topology_can_overlap({'a': (1,), 'b': ('z',), 'c': (1, 'q')})

    # a bare '_path' writes only under its base, even nested in a dict
    assert not topology_can_overlap(
        {'a': {'_path': ('x',)}, 'b': ('y',)})
    assert not topology_can_overlap(
        {'a': {'v': {'_path': ('x',)}, 'w': ('y',)}})
    assert topology_can_overlap(
        {'a': {'_path': ('x',)}, 'b': ('x', 'z')})
    # explicit '_path' routes and globs aren't resolved statically
    assert topology_can_overlap({'a': {'_path': ('x',), 'v': ('v',)}})
    assert topology_can_overlap({'*': ('cells',)})

    # large topologies are checked in roughly linear time
    many: Topology = {f'p{i}': ('cell', f's{i}') for i in range(20000)}
    assert not topology_can_overlap(many)
    many['last'] = ('cell', 's123', 'deeper')
    assert topology_can_overlap(many)


def test_topology_split_plan() -> None:
    """Consecutive non-overlapping ports share a group; a port starts a new
    group when it overlaps the current one, and ports that can't be
    resolved statically or that overlap themselves stand alone."""
    path_routes = {'_path': ('p',), 'r': ('q',)}
    self_overlap = {'u': ('s',), 'v': ('s',)}
    topology: Topology = {
        'a': ('x',),
        'b': ('y',),
        'c': ('x', 'deep'),  # nested in 'a': new group
        'd': ('z',),
        '*': ('cells',),
        'e': path_routes,
        'f': self_overlap,
        'g': ('w',),
        'h': ('x',),  # 'a' is in an earlier group, so no new group
    }
    assert topology_split_plan(topology) == [
        (True, {'a': ('x',), 'b': ('y',)}),
        (True, {'c': ('x', 'deep'), 'd': ('z',)}),
        (False, {'*': ('cells',)}),
        (False, {'e': path_routes}),
        (False, {'f': self_overlap}),
        (True, {'g': ('w',), 'h': ('x',)}),
    ]
    assert topology_split_plan({'a': ('x',), 'b': ('y',)}) is None
    assert topology_split_plan({}) is None

    # one overlap among many ports costs one extra update, not one each
    many: Topology = {f'p{i}': (f's{i}',) for i in range(50)}
    many['p1'] = ('s0',)
    update = {port: {'v': 1} for port in many}
    inverted = invert_topology(
        update, (('proc',), many, topology_split_plan(many)))
    assert len(inverted) == 2
    assert inverted[0] == {'s0': {'v': 1}}
    assert inverted[1]['s0'] == {'v': 1}
    assert len(inverted[1]) == 49


class BarePathProcess(Process):
    """Ports wired with bare ``_path`` topologies, one holding a leaf."""
    name = 'bare_path_process'

    def ports_schema(self) -> Schema:
        return {
            'trigger': {'_default': False, '_updater': 'set'},
            'cell': {
                'x': {'_default': 0},
                'y': {'_default': 0}},
            'other': {'z': {'_default': 0}}}

    def next_update(
            self,
            timestep: Union[float, int],
            states: State) -> Update:
        return {
            'trigger': True,
            'cell': {'x': 1, 'y': 2},
            'other': {'z': 3}}


def test_bare_path_fast_path() -> None:
    """Bare ``_path`` ports with distinct bases take the single-pass fast
    path, and give the same result as when they overlap and are split."""
    topology: Dict[str, Any] = {
        'trigger': {'_path': ('trig',)},
        'cell': {'_path': ('cell',)},
        'other': ('elsewhere',)}
    exp = Engine(
        processes={'proc': BarePathProcess({})},
        topology={'proc': topology},
        initial_state={'trig': False, 'cell': {'x': 0, 'y': 0}})
    exp.update(1)
    assert exp.state.get_path(('proc',)).invert_can_overlap is False
    assert exp.state.get_path(('trig',)).get_value() is True
    assert exp.state.get_path(('cell',)).get_value() == {'x': 1, 'y': 2}
    assert exp.state.get_path(('elsewhere',)).get_value() == {'z': 3}

    # point 'other' into the 'cell' base: both writes to cell/z apply
    overlapping = dict(topology, other=('cell',))
    exp = Engine(
        processes={'proc': BarePathProcess({})},
        topology={'proc': overlapping},
        initial_state={'trig': False, 'cell': {'x': 0, 'y': 0, 'z': 0}})
    exp.update(1)
    assert exp.state.get_path(('proc',)).invert_can_overlap is True
    assert exp.state.get_path(('cell',)).get_value() == {
        'x': 1, 'y': 2, 'z': 3}


def test_path_defaults_inverted_together() -> None:
    """Unlisted keys of a split ``_path`` port go to distinct children of
    its base, so they're inverted as one update after the explicit routes
    instead of one update each -- and an explicit route aliasing a default
    key still gets both contributions."""
    update = {
        'port': {'base_var': 1, 'near_var': 2, 'extra': 4, 'far_var': 3}}
    topology: Topology = {
        'port': {
            '_path': ('base',),
            'near_var': ('near',),
            'far_var': ('..', 'far')}}
    inverted = invert_topology(
        update, (('proc',), topology, topology_split_plan(topology)))
    # one update per explicit route, then one for all the defaults
    assert inverted == [
        {'base': {'near': 2}},
        {'far': 3},
        {'base': {'base_var': 1, 'extra': 4}}]

    # 'near_var' is routed onto base/base_var, the default key's store
    process = PathRewireProcess({})
    exp = Engine(
        processes={'proc': process},
        topology={
            'proc': {
                'port': {
                    '_path': ('base',),
                    'near_var': ('base_var',),
                    'far_var': ('..', 'far')}}},
        initial_state={'base': {'base_var': 0}, 'far': 0})
    exp.update(1)
    assert exp.state.get_path(('base', 'base_var')).get_value() == 3


def test_emit_config() -> None:
    # test alternate emit options
    merge_port = MergePort({})
    network = merge_port.generate()
    exp1 = Engine(
        processes=network['processes'],
        topology=network['topology'],
        emit_topology=False,
        emit_processes=True,
        emit_config=True,
        progress_bar=True,
        emit_step=2,
    )

    exp1.update(10)


def test_complex_topology() -> None:
    # make the experiment
    outer_path = ('universe', 'agent')
    pq = PoQo({})
    pq_composite = pq.generate(path=outer_path)
    pq_composite.pop('_schema')
    experiment = Engine(composite=pq_composite)

    # get the initial state
    initial_state = experiment.state.get_value()
    print('time 0:')
    pp(initial_state)

    # simulate for 1 second
    experiment.update(1)

    next_state = experiment.state.get_value()
    print('time 1:')
    pp(next_state)

    # pull out the agent state
    initial_agent_state = initial_state['universe']['agent']
    agent_state = next_state['universe']['agent']

    assert agent_state['aaa']['a1'] == initial_agent_state['aaa']['a1'] + 1
    assert agent_state['aaa']['x'] == initial_agent_state['aaa']['x'] - 9
    assert agent_state['ccc']['a3'] == initial_agent_state['ccc']['a3'] + 1


def test_parallel() -> None:
    proton = _make_proton(parallel=True)
    experiment = Engine(**proton)

    log.debug(pf(experiment.state.get_config(True)))

    experiment.update(10.0)

    log.debug(pf(experiment.state.get_config(True)))
    log.debug(pf(experiment.state.divide_value()))

    experiment.end()


def test_depth() -> None:
    nested = {
        'A': {
            'AA': 5,
            'AB': {
                'ABC': 11}},
        'B': {
            'BA': 6}}

    dissected = hierarchy_depth(nested)
    assert len(dissected) == 3
    assert dissected[('A', 'AB', 'ABC')] == 11


def test_sine() -> None:
    sine = Sine()
    print(sine.next_update(0.25 / 440.0, {
        'frequency': 440.0,
        'amplitude': 0.1,
        'phase': 1.5}))


def test_units() -> None:
    class UnitsMicrometer(Process):
        name = 'units_micrometer'

        def ports_schema(self) -> Schema:
            return {
                'A': {
                    'a': {
                        '_default': 0 * units.um,
                        '_emit': True},
                    'b': {
                        '_default': 'string b',
                        '_emit': True,
                    },
                    'c': {
                        '_default': 0,
                        '_emit': True,
                    }
                }
            }

        def next_update(
                self, timestep: Union[float, int], states: State) -> Update:
            return {
                'A': {
                    'a': 1 * units.um,
                    'c': 1,
                }}

    class UnitsMillimeter(Process):
        name = 'units_millimeter'

        def ports_schema(self) -> Schema:
            return {
                'A': {
                    'a': {
                        # '_default': 0 * units.mm,
                        '_emit': True}}}

        def next_update(
                self, timestep: Union[float, int], states: State) -> Update:
            return {
                'A': {'a': 1 * units.mm}}

    class MultiUnits(Composer):
        name = 'multi_units_composer'

        def generate_processes(
                self,
                config: Optional[dict]) -> Dict[str, Any]:
            return {
                'units_micrometer':
                    UnitsMicrometer({}),
                'units_millimeter':
                    UnitsMillimeter({})}

        def generate_topology(self, config: Optional[dict]) -> Topology:
            return {
                'units_micrometer': {
                    'A': ('aaa',)},
                'units_millimeter': {
                    'A': ('aaa',)}}

    # run experiment
    multi_unit = MultiUnits({})
    network = multi_unit.generate()
    exp = Engine(**{
        'processes': network['processes'],
        'topology': network['topology']})

    exp.update(5)
    timeseries = exp.emitter.get_timeseries()
    print('TIMESERIES')
    pp(timeseries)

    data = exp.emitter.get_data()
    print('DATA')
    pp(data)

    data_deserialized = exp.emitter.get_data_deserialized()
    print('DESERIALIZED')
    pp(data_deserialized)

    data_unitless = exp.emitter.get_data_unitless()
    print('UNITLESS')
    pp(data_unitless)

    query = [('aaa', 'a'), ('aaa', 'c')]
    query_data = exp.emitter.get_data(query)
    print('QUERY DATA')
    pp(query_data)


def test_custom_divider() -> None:
    """ToyDividerProcess has a custom `split_divider`"""
    agent_id = '1'
    composer = ToyDivider({
        'agent_id': agent_id,
        'divider': {
            'x_division_threshold': 3,
        }
    })
    composite = composer.generate(path=('agents', agent_id))

    experiment = Engine(
        processes=composite.processes,
        steps=composite.steps,
        flow=composite.flow,
        topology=composite.topology,
    )

    experiment.update(8)
    data = experiment.emitter.get_data()

    expected_data = {
        0.0: {'agents': {'1': {'variable': {'x': 0, '2x': 0}}}},
        1.0: {'agents': {'1': {'variable': {'x': 1, '2x': 2}}}},
        2.0: {'agents': {'1': {'variable': {'x': 2, '2x': 4}}}},
        3.0: {'agents': {'1': {'variable': {'x': 3, '2x': 6}}}},
        4.0: {'agents': {'1': {'variable': {'x': 4, '2x': 8}}}},
        5.0: {'agents': {'10': {'variable': {'x': 2, '2x': 4}},
                         '11': {'variable': {'x': 2, '2x': 4}}}},
        6.0: {'agents': {'10': {'variable': {'x': 3, '2x': 6}},
                         '11': {'variable': {'x': 3, '2x': 6}}}},
        7.0: {'agents': {'10': {'variable': {'x': 4, '2x': 8}},
                         '11': {'variable': {'x': 4, '2x': 8}}}},
        8.0: {'agents': {'100': {'variable': {'x': 2, '2x': 4}},
                         '101': {'variable': {'x': 2, '2x': 4}},
                         '110': {'variable': {'x': 2, '2x': 4}},
                         '111': {'variable': {'x': 2, '2x': 4}}}}
    }
    assert data == expected_data


class TestStepGraph:

    @staticmethod
    def test_step_graph_execution_layers() -> None:
        tg = _StepGraph()
        tg.add(('a',), [])
        tg.add(('b',), [])
        tg.add(('c',), [('a',), ('b',)])
        tg.add(('d',), [('c',)])

        layers = list(tg.get_execution_layers())
        expected_layers = [
            [('a',), ('b',)],
            [('c',)],
            [('d',)],
        ]
        assert layers == expected_layers

    @staticmethod
    def test_step_graph_sequential() -> None:
        tg = _StepGraph()
        tg.add(('a',), [])
        tg.add(('b',), [])
        tg.add_sequential(('c',))

        layers = list(tg.get_execution_layers())
        expected_layers = [
            [('c',)],
            [('a',), ('b',)],
        ]
        assert layers == expected_layers

    @staticmethod
    def test_step_graph_removal() -> None:
        tg = _StepGraph()
        tg.add(('a',), [])
        tg.add(('b',), [('a',)])
        tg.add(('c',), [('b',)])

        tg.remove(('a',))
        layers = list(tg.get_execution_layers())

        assert not layers


def test_runtime_order() -> None:
    class RuntimeOrderProcess(Process):

        def ports_schema(self) -> Schema:
            return {
                'store': {
                    'var': {
                        '_default': 0
                    }
                }
            }

        def next_update(self, timestep: float, states: State) -> Update:
            _ = states
            self.parameters['execution_log'].append(self.name)
            return {}

    class RuntimeOrderStep(Step):

        def ports_schema(self) -> Schema:
            return {
                'store': {
                    'var': {
                        '_default': 0
                    }
                }
            }

        def next_update(self, timestep: float, states: State) -> Update:
            _ = states
            self.parameters['execution_log'].append(self.name)
            return {}

    class RuntimeOrderDeriver(Deriver):

        def ports_schema(self) -> Schema:
            return {
                'store': {
                    'var': {
                        '_default': 0
                    }
                }
            }

        def next_update(self, timestep: float, states: State) -> Update:
            _ = states
            self.parameters['execution_log'].append(self.name)
            return {}

    class RuntimeOrderComposer(Composer):

        def generate_processes(
                self, config: Optional[dict]) -> Dict[str, Any]:
            config = cast(dict, config or {})
            proc1 = RuntimeOrderProcess({
                'name': 'process1',
                'time_step': 1,
                'execution_log': config['execution_log'],
            })
            proc2 = RuntimeOrderProcess({
                'name': 'process2',
                'time_step': 2,
                'execution_log': config['execution_log'],
            })
            deriver = RuntimeOrderDeriver({
                'name': 'deriver',
                'execution_log': config['execution_log'],
            })
            return {
                'p1': proc1,
                'p2': proc2,
                'd': deriver,
            }

        def generate_steps(self, config: Optional[dict]) -> Steps:
            config = config or {}
            step1 = RuntimeOrderStep({
                'name': 'step1',
                'execution_log': config['execution_log'],
            })
            step2 = RuntimeOrderStep({
                'name': 'step2',
                'execution_log': config['execution_log'],
            })
            step3 = RuntimeOrderStep({
                'name': 'step3',
                'execution_log': config['execution_log'],
            })
            return {
                's1': step1,
                's2': step2,
                's3': step3,
            }

        def generate_flow(self, config: Optional[dict]) -> Steps:
            config = config or {}
            return {
                's1': [],
                's2': [('s1',)],
                's3': [('s1',)],
            }

        def generate_topology(self, config: Optional[dict]) -> Topology:
            return {
                'p1': {
                    'store': ('store',),
                },
                'p2': {
                    'store': ('store',),
                },
                'd': {
                    'store': ('store',),
                },
                's1': {
                    'store': ('store',),
                },
                's2': {
                    'store': ('store',),
                },
                's3': {
                    'store': ('store',),
                },
            }

    execution_log: List[str] = []
    composer = RuntimeOrderComposer()
    composite = composer.generate({'execution_log': execution_log})
    experiment = Engine(
        processes=composite.processes,
        steps=composite.steps,
        flow=composite.flow,
        topology=composite.topology,
    )
    experiment.update(4)
    expected_log = [
        ('deriver', 'step1'),
        {'step2', 'step3'},
        {'process1', 'process2'},
        ('deriver', 'step1'),
        {'step2', 'step3'},
        {'process1'},
        ('deriver', 'step1'),
        {'step2', 'step3'},
        {'process1', 'process2'},
        ('deriver', 'step1'),
        {'step2', 'step3'},
        {'process1'},
        ('deriver', 'step1'),
        {'step2', 'step3'},
    ]
    for expected_group in expected_log:
        num = len(expected_group)
        group = execution_log[0:num]
        execution_log = execution_log[num:]
        if isinstance(expected_group, tuple):
            assert tuple(group) == expected_group
        elif isinstance(expected_group, set):
            assert set(group) == expected_group


def get_toy_transport_in_env_composite(
        agent_id: Any = '0',
        transport_config: Optional[dict] = None,
        environment_config: Optional[dict] = None,
) -> Composite:
    processes = {
        'agents': {
            agent_id: {
                'transport': ToyTransport(transport_config)}},
        'environment': ToyEnvironment(environment_config)}
    topology = {
        'environment': {
            'agents': {
                '_path': ('agents',),
                '*': {
                    'external': ('external', 'GLC')}}},
        'agents': {
            agent_id: {
                'transport': {
                    'internal': ('internal',),
                    'external': ('external',)}}}}
    return Composite({
        'processes': processes,
        'topology': topology,
    })


def test_glob_schema() -> None:
    composite = get_toy_transport_in_env_composite()
    experiment = Engine(
        processes=composite.processes,
        topology=composite.topology)
    experiment.update(10)

    # declare processes in reverse order
    processes_reverse = {
        'environment': ToyEnvironment(),
        'agents': {'0': {'transport': ToyTransport()}}}

    experiment_reverse = Engine(
        processes=processes_reverse,
        topology=composite.topology)
    experiment_reverse.update(10)


def get_env_view_composite(agent_id: Any = '1') -> Composite:
    agent_composer = ToyDivider({
        'agent_id': agent_id,
        'divider': {
            'x_division_threshold': 3,
        }
    })
    composite = agent_composer.generate(path=('agents', agent_id))

    environment_process = {
        'environment': make_logging_process(ToyEnvironment)()
    }
    environment_topology: Topology = {
        'environment': {
            'agents': {
                '_path': ('agents',),
                '*': {
                    'external': ('external', 'GLC')
                }
            },
            'log_update': ('log_update',),
        }
    }

    # combine the environment and agent
    composite.merge(
        processes=environment_process,
        topology=environment_topology,
    )
    return composite


def test_environment_view_with_division() -> None:
    composite = get_env_view_composite()
    experiment = Engine(
        processes=composite.processes,
        topology=composite.topology)
    experiment.update(10)
    data = experiment.emitter.get_data()

    # confirm that the environment sees the new agents.
    once_different = False
    for state in data.values():
        agent_ids = set(state['agents'].keys())
        env_agents = set(state['log_update'].get('agents', {}).keys())
        if env_agents != agent_ids:
            if not once_different:
                once_different = True
            else:
                # the values have been different for more than one update
                raise ValueError(
                    f'environment sees {env_agents} instead of {agent_ids}')
        else:
            once_different = False


class AddDelete(Process):
    def ports_schema(self) -> Schema:
        return {
            'sub_stores': {
                '*': {
                    '_default': 0,
                    '_emit': True
                }
            },
            'expected': {
                '_default': [],
                '_updater': 'set',
            }
        }

    def next_update(
            self, timestep: Union[float, int], states: State) -> Update:
        sub_stores = set(states['sub_stores'].keys())
        expected = set(states['expected'])
        assert sub_stores == expected, "stores don't match expected"

        # delete current stores, and add the same number of stores
        sub_stores_update: dict = {
            '_delete': [],
            '_add': []}
        new_sub_stores = []
        for store_id in sub_stores:
            sub_stores_update['_delete'].append(store_id)
            new_id = str(random.randint(0, 2 ** 63))
            new_sub_stores.append(new_id)
            new_store = {'key': new_id, 'state': 1}
            sub_stores_update['_add'].append(new_store)

        return {
            'sub_stores': sub_stores_update,
            'expected': new_sub_stores,
        }


def test_add_delete() -> None:
    process = AddDelete()
    topology = {
        'sub_stores': ('sub_stores',),
        'expected': ('expected',)}

    # initial state
    n_initial = 10
    initial_substores = [
        str(random.randint(0, 2**63)) for _ in range(n_initial)]
    initial_state = {
        'sub_stores': {
            sub_store: 1
            for sub_store in initial_substores
        },
        'expected': initial_substores,
    }

    experiment = Engine(
        processes={'process': process},
        topology={'process': topology},
        initial_state=initial_state,
    )
    experiment.update(10)

    # assert that no overlapping sub store between time steps.
    # All sub stores should get deleted, and new sub stores added
    data = experiment.emitter.get_data()
    times = list(data.keys())
    n_times = len(times)
    for t_index in range(n_times):
        if t_index < n_times - 1:
            current_time = times[t_index]
            next_time = times[t_index + 1]
            current_ids = set(data[current_time]['sub_stores'].keys())
            next_ids = set(data[next_time]['sub_stores'].keys())
            assert len(set(current_ids).intersection(set(next_ids))) == 0


def test_hyperdivision(profile: bool = True) -> None:
    total_time = 10
    n_agents = 100
    division_thresholds = [3, 4, 5, 6, 7]  # what values of x triggers division?

    # initialize agent composer
    agent_composer = ToyDivider()

    # make the composite
    composite = Composite()
    agent_ids = [str(agent_idx) for agent_idx in range(n_agents)]
    for agent_id in agent_ids:
        divider_config = {
            'divider': {
                    'x_division_threshold': random.choice(division_thresholds),
                }}
        agent_composite = agent_composer.generate(
            config={
                'agent_id': agent_id,
                **divider_config,
            },
            path=('agents', agent_id))
        composite.merge(agent_composite)

    # add an environment
    environment_process: Processes = {'environment': ToyEnvironment()}
    environment_topology: Topology = {
        'environment': {
            'agents': {
                '_path': ('agents',),
                '*': {
                    'external': ('external', 'GLC')
                }
            },
        }
    }

    # combine the environment and agent
    composite.merge(
        processes=environment_process,
        topology=environment_topology,
    )

    # make the sim, run the sim, retrieve the data
    experiment = Engine(
        processes=composite.processes,
        steps=composite.steps,
        flow=composite.flow,
        topology=composite.topology,
        profile=profile,
    )
    experiment.update(total_time)
    experiment.end()
    data = experiment.emitter.get_data()

    print(f"n agents initial: {n_agents}")
    print(f"n agents final: {len(data[total_time]['agents'].keys())}")
    assert len(data[total_time]['agents'].keys()) > n_agents

    if profile:
        stats = experiment.stats
        stats.strip_dirs().sort_stats(  # type: ignore
            'cumulative', 'cumtime').print_stats(20)

        # make sure view_values is fast
        stats_view_values = stats.get_print_list(  # type: ignore
            ('view_values',))[1]
        view_values_times = stats.stats[  # type: ignore
            stats_view_values[0]][3]
        total_runtime = stats.total_tt  # type: ignore
        assert view_values_times < 0.1 * total_runtime

def test_output_port() -> None:
    a_default = 1
    b_default = 2

    class InputOutput(Process):
        def ports_schema(self) -> Schema:
            return {
                'input': {
                    'A': {
                        '_default': a_default,
                        '_emit': True,
                    }
                },
                'output': {
                    '_output': True,
                    'B': {
                        '_default': b_default,
                        '_emit': True,
                    }
                }
            }
        def next_update(
                self,
                timestep: Union[float, int],
                states: State) -> Update:
            assert not states['output'], 'outputs should be masked'
            return {}

    total_time = 10
    composite = InputOutput().generate()
    sim = Engine(
        processes=composite['processes'],
        topology=composite['topology']
    )
    sim.update(total_time)
    data = sim.emitter.get_timeseries()
    assert data['input']['A'] == [a_default for _ in range(total_time + 1)]
    assert data['output']['B'] == [b_default for _ in range(total_time + 1)]


def test_engine_run_for() -> None:
    total_time = 10.0
    time_interval = 1.0
    timestep1 = 0.75
    timestep2 = 1.25

    topo = {
        'external': ('external',),
        'internal': ('internal',),
    }
    composite = Composite({
        'processes': {
            'process1': ToyTransport({'time_step': timestep1}),
            'process2': ToyTransport({'time_step': timestep2}),
        },
        'topology': {
            'process1': topo,
            'process2': topo,
        }
    })
    initial_state = {
        'external': {'GLC': 100},
    }

    sim = Engine(
        processes=composite.processes,
        topology=composite.topology,
        initial_state=initial_state)

    time = 0.0
    while sim.global_time < total_time:
        if time + time_interval >= total_time:
            # If this is the last iteration of the simulation loop,
            # force processes to complete.
            sim.run_for(time_interval, force_complete=True)
        else:
            sim.run_for(time_interval)
        time += time_interval
        assert sim.global_time == time

        # check that the front has advanced correctly
        front = sim.front
        for path, advance in front.items():
            expected_time = 0.0
            if time >= total_time:
                expected_time = total_time
            elif path[0] == 'process1':
                expected_time = int(time / timestep1) * timestep1
            elif path[0] == 'process2':
                expected_time = int(time / timestep2) * timestep2
            assert advance['time'] == expected_time, \
                f"front time {advance['time']} " \
                f"is not expected {expected_time}"

    final_time = time
    assert sim.global_time == final_time

    front = sim.front
    for path, advance in front.items():
        assert advance['time'] == sim.global_time, \
            f"process at path {path} did not complete"


def test_set_branch_emit() -> None:
    run_time = 5
    # get the composer
    composer = PoQo({})

    # turn on emits
    composite = composer.generate()
    exp = Engine(
        composite=composite,
        store_schema={'_emit': True}
    )
    exp.update(run_time)
    data = exp.emitter.get_data()
    assert data[run_time]['bbb'] != {}, 'this emit should be on'
    print(pf(data))

    # turn off emits
    composite = composer.generate()
    exp = Engine(
        composite=composite,
        store_schema={'_emit': False}
    )
    exp.update(run_time)
    data = exp.emitter.get_data()
    assert data[run_time]['bbb'] == {}, 'this emit should be off'
    print(pf(data))

    # selectively turn on emits
    composite = composer.generate()
    exp = Engine(
        composite=composite,
        store_schema={'bbb': {'e2': {'_emit': True}}},
    )
    exp.update(run_time)
    data = exp.emitter.get_data()
    assert data[run_time]['bbb']['e2'] != {}, 'this emit should be on'
    assert data[run_time]['ccc'] == {}, 'this emit should be off'
    print(pf(data))


def test_add_new_state() -> None:
    agent_id = '1'
    composite = get_toy_transport_in_env_composite(agent_id=agent_id)
    new_schema = {
        'agents': {
            agent_id: {
                'extra': {'_emit': True, '_value': 1.0}}}}
    experiment = Engine(
        processes=composite.processes,
        topology=composite.topology,
        store_schema=new_schema,
    )

    assert experiment.state['agents', agent_id, 'extra'].get_value() == 1.0

    total_time = 5
    experiment.update(total_time)
    timeseries = experiment.emitter.get_timeseries()
    assert len(timeseries['agents'][agent_id]['extra']) == total_time + 1


def test_floating_point_timesteps() -> None:
    agent_id = '1'
    transport_timestep = 0.1
    composite = get_toy_transport_in_env_composite(
        agent_id=agent_id,
        transport_config={'time_step': transport_timestep}
    )
    sim = Engine(
        processes=composite.processes,
        topology=composite.topology,
        global_time_precision=5
    )
    sim.update(1)
    data = sim.emitter.get_data()
    print(pf(data))
    for t in data.keys():
        ntimes = round(t / transport_timestep, 5)
        expected_t = round(ntimes * transport_timestep, 5)
        assert t == expected_t, f'bad timestep {t}'


def test_move_update() -> None:
    agents_in_1 = ['a1', 'a2']
    agents_in_2 = ['a3', 'a4']
    agents_in_3 = ['a5', 'a6']

    composite = Composite({
        'processes': {
            'move': MoveProcess({'timestep': 2}),
            'store1': {
                agent_id: {
                    'toy': ToyProcess({'timestep': 1})
                } for agent_id in agents_in_1},
        },
        'topology': {
            'move': {
                '1': ('store1',),
                '2': ('store2',),
                '3': ('store3',),
            },
            'store1': {
                agent_id: {
                    'toy': {
                        'port1': ('store_A',),
                        'port2': ('store_B',)
                    }
                } for agent_id in agents_in_1},
        }})

    state = {
        'store1': {
            agent_id: {
                'store_A': {'var_a': 0.1, 'var_b': 0.2},
                'store_B': {'var_a': 0.3, 'var_b': 0.4},
            } for agent_id in agents_in_1
        },
        'store2': {
            agent_id: {} for agent_id in agents_in_2
        },
        'store3': {
            agent_id: {} for agent_id in agents_in_3
        },
    }

    sim = Engine(
        processes=composite.processes,
        topology=composite.topology,
        initial_state=state,
    )

    sim.update(4)
    data = sim.emitter.get_data()

    # it should move at t=2, t=4
    assert list(data[4]['store1'].keys()) == agents_in_2
    assert list(data[4]['store2'].keys()) == agents_in_3
    assert list(data[4]['store3'].keys()) == agents_in_1
    assert 'var_a' in data[4]['store3'][agents_in_1[0]]['store_A'], \
        'the agent initialized in store1 should have ' \
        'carried its internal states generated by ToyProcess'



engine_tests = {
    '0': test_recursive_store,
    '1': test_topology_ports,
    '2': test_timescales,
    '3': test_2_store_1_port,
    '4': test_multi_port_merge,
    '5': test_emit_config,
    '6': test_complex_topology,
    '7': test_parallel,
    '8': test_depth,
    '9': test_sine,
    '10': test_units,
    '11': test_custom_divider,
    '12': test_runtime_order,
    '13': test_glob_schema,
    '14': test_environment_view_with_division,
    '15': test_add_delete,
    '16': test_hyperdivision,
    '17': test_output_port,
    '18': test_engine_run_for,
    '19': test_set_branch_emit,
    '20': test_add_new_state,
    '21': test_floating_point_timesteps,
    '22': test_move_update,
    '23': test_dict_topology_multi_port,
    '24': test_multi_port_initial_state,
    '25': test_multi_port_initial_state_direct_leaf,
    '26': test_multi_port_dict_values,
    '27': test_mixed_topology_multi_port,
    '28': test_direct_leaf_multi_port,
    '29': test_partial_update_multi_port,
    '30': test_sibling_var_multi_port,
    '31': test_nested_subtree_multi_port,
    '32': test_glob_root_multi_port,
    '33': test_glob_plus_direct_port,
    '34': test_intra_port_scalar_overlap,
    '35': test_intra_port_dict_overlap,
    '36': test_path_route_overlap,
    '37': test_path_directive_not_split,
    '38': test_path_base_rooted_routes,
    '39': test_one_off_updater_override,
    '40': test_invert_cache_invalidated_on_rewire,
    '41': test_same_leaf_overlap_ordering,
    '42': test_overlapping_delete_not_duplicated_or_dropped,
    '43': test_topology_edits_invalidate_cache,
    '44': test_topology_can_overlap,
    '47': test_topology_split_plan,
    '45': test_bare_path_fast_path,
    '46': test_path_defaults_inverted_together,
}


# python vivarium/experiments/engine_tests.py -n [test number]
if __name__ == '__main__':
    run_library_cli(engine_tests)
