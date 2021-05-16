from vivarium.composites.toys import Qo, ToyProcess, ToyComposer
from vivarium.core.store import Store, topology_path


def test_store1():
    composer = ToyComposer({
        'A':  {'name': 'process1'},
        'B': {'name': 'process2'}})
    composite = composer.generate()
    process = ToyProcess({'name': 'process3'})

    store = composite.generate_store()
    store.insert({
        'key': tuple(),
        'processes': {'process3': process},
        'topology': {
            'process3': {
                'A': {'_path': ('ccc',), 'a': ('d',)},
                'B': ('aaa',)
            }
        },
        'initial_state': {}})

    path, remaining = topology_path(store.get_topology(), ('process1', 'A', 'b'))

    import ipdb; ipdb.set_trace()

    # connect process1's port A to the store at process3's port A
    store['process1']['A'] = store['process3']['A']
    # store['process1','A'] = store['process3','A']

    # connect process2's port B to store ccc
    store['process2']['B'] = store['ccc']

    # replace a process
    store['process4'] = ToyProcess({'name': 'process4'})
    # store['process4']['A'] = Store()  # TODO: this should give an error

    # connect port A to a new store ddd
    """
    topology before: 
    process4: {
        'A': ('aaa',),
        'B': ('bbb',),}
        
    topology after:
    process4: {
        'A': ('ddd',),
        'B': ('bbb',),}
    """

    store['ddd'] = Store({})
    store['process4']['A'] = store['ddd']

    # set a value through a port
    store['process1']['A']['a'] = 2

    # replace a process with different ports entirely
    store['process1'] = Qo({})

    # replace a process with a subset of the same ports




if __name__ == '__main__':
    test_store1()
