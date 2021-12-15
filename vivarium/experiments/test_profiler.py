import time

from vivarium.core.engine import Engine
from vivarium.core.process import Process
from vivarium.core.types import Schema, State, Update


class ProcessA(Process):

    defaults = {
        'sleep': 0.2,
    }

    def ports_schema(self) -> Schema:
        return {}

    def next_update(self, timestep: float, states: State) -> Update:
        time.sleep(self.parameters['sleep'])
        return {}


class ProcessB(Process):

    defaults = {
        'sleep': 0.1,
        '_parallel': True,
    }

    def ports_schema(self) -> Schema:
        return {}

    def next_update(self, timestep: float, states: State) -> Update:
        time.sleep(self.parameters['sleep'])
        return {}


def test_profiler() -> None:
    engine = Engine(
        processes={
            'processA': ProcessA(),
            'processB': ProcessB(),
        },
        topology={
            'processA': {},
            'processB': {},
        },
        profile=True,
    )
    engine.update(3)
    engine.end()
    assert engine.stats is not None
    stats = engine.stats.strip_dirs()
    process_a_runtime = stats.stats[  # type: ignore
        ('test_profiler.py', 17, 'next_update')][3]
    process_b_runtime = stats.stats[  # type: ignore
        ('test_profiler.py', 32, 'next_update')][3]

    assert 0.6 <= process_a_runtime <= 0.7
    assert 0.3 <= process_b_runtime <= 0.4
