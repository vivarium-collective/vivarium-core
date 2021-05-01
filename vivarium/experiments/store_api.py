from vivarium.composites.toys import ToyProcess, ToyComposer

def test_store_insert():
    composer = ToyComposer()
    composite = composer.generate()
    process = ToyProcess({'name': 'C'})

    store = composite.generate_store()
    store.insert({
        'key': tuple(),
        'processes': {'C': process},
        'topology': {
            'C': {
                'A': ('ccc',),
                'B': ('aaa',)
            }
        },
        'initial_state': {}})

    # store.insert({
    #     'key': tuple(),
    #     'processes': {'C': process},
    #     'topology': {
    #         'C[A]': 'B[B]',
    #         'C[B]': 'A[A]',
    #     },
    #     'initial_state': {}})

    # store['C'] = process
    # store.get(('C', 'A', 'b')) # --> store node whose value maybe a process

    # store['C']
    # store['C']['A'] # store that holds process A, not process A
    # store['C']['A'] = store['B']['B']
    # store['C']['A']['b'] = store['B']['B']['a']
    # store['C']['B'] = store['A']['A']

    # store['agents']['1'].divide()

    topology = store.get_topology()
    assert 'C' in topology
    
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    test_store_insert()
