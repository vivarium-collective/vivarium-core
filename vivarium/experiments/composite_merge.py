from vivarium.composites.toys import ExchangeA
from vivarium.core.process import Composite



def test_multi_composite():

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
    print(f"composite3 processes: {composite3['processes']}")
    print(f"composite3 topology: {composite3['topology']}")

    composite1.merge(composite=composite2, path=('agents', '1',))

    composite4 = Composite()
    print(f"composite4 processes: {composite4['processes']}")
    print(f"composite4 topology: {composite4['topology']}")





if __name__ == '__main__':
    test_multi_composite()
