from vivarium.composites.toys import ExchangeA, ToyComposer
from vivarium.core.composer import Composite
from vivarium.core.engine import pf
from vivarium.core.control import run_library_cli
from vivarium.experiments.engine_tests import get_toy_transport_in_env_composite


def test_multi_composite() -> None:

    composite1 = Composite()

    processes = {
        'A': ExchangeA()
    }
    topology = {
        'A': {
            'internal': ('in',),
            'external': ('out',)}}
    composite2 = Composite({
            'processes': processes,
            'topology': topology,
        })

    composite3 = Composite()
    assert not composite3['processes']
    assert not composite3['topology']
    print(f"composite3 processes: {composite3['processes']}")
    print(f"composite3 topology: {composite3['topology']}")

    composite1.merge(composite=composite2, path=('agents', '1',))

    composite4 = Composite()
    assert not composite4['processes']
    assert not composite4['topology']
    print(f"composite4 processes: {composite4['processes']}")
    print(f"composite4 topology: {composite4['topology']}")


def test_store_composite():
    composite = get_toy_transport_in_env_composite()
    agent_composite = composite['agents', '0']

    initial_state = {
        'agents': {'0': {'external': {'GLC': 10.0}}}}

    composite.run_for(10)
    data = composite.get_data()
    print(pf(data))

    import ipdb; ipdb.set_trace()



test_library = {
    '1': test_multi_composite,
    '2': test_store_composite,
}


# python vivarium/experiments/composite_merge.py -n [test number]
if __name__ == '__main__':
    run_library_cli(test_library)
