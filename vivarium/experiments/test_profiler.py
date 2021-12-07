import time

from vivarium.core.engine import Engine
from vivarium.core.process import Process


class ProcessA(Process):

    defaults = {
        'sleep': 2,
    }

    def ports_schema(self):
        return {}

    def next_update(self, timestep, states):
        time.sleep(self.parameters['sleep'])
        return {}


class ProcessB(Process):

    defaults = {
        'sleep': 1,
        '_parallel': True,
    }

    def ports_schema(self):
        return {}

    def next_update(self, timestep, states):
        time.sleep(self.parameters['sleep'])
        return {}


def test_profiler():
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
    stats = engine.stats.strip_dirs()
    process_a_runtime = stats.stats[
        ('test_profiler.py', 16, 'next_update')][3]
    process_b_runtime = stats.stats[
        ('test_profiler.py', 31, 'next_update')][3]

    assert 6 <= process_a_runtime <= 6.1
    assert 3 <= process_b_runtime <= 3.1
