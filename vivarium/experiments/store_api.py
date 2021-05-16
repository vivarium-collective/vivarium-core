from vivarium.composites.toys import Qo, ToyProcess, ToyComposer
from vivarium.core.store import Store
from vivarium.core.process import Process
from vivarium.core.experiment import Experiment


def get_toy_store():
    """get a store to test the api with"""
    composer = ToyComposer({
        'A':  {'name': 'process1'},
        'B': {'name': 'process2'}})
    composite = composer.generate()
    store = composite.generate_store()
    return store


def test_insert_process():
    """Test Store.insert by adding a new process

    Return:
        the toy store with a process3
    """
    store = get_toy_store()
    process = ToyProcess({'name': 'process3'})
    store.insert({
        'key': tuple(),
        'processes': {'process3': process},
        'topology': {
            'process3': {
                'A': ('ccc',),
                'B': ('aaa',)}},
        'initial_state': {}})

    assert isinstance(store['process3'].value, Process), 'process3 not inserted successfully'
    return store


def test_rewire_ports():
    """connect a process' ports to different store"""
    store = test_insert_process()

    # connect process1's port A to the store at process3's port A
    store['process1']['A'] = store['process3']['A']
    assert store['process1']['A'] == store['process3']['A']
    import ipdb; ipdb.set_trace()


    # this should give the same result
    store = get_toy_store()
    store['process1', 'A'] = store['process3', 'A']
    assert store['process1']['A'] == store['process3']['A']

    # connect process2's port B to store aaa
    store['process2']['B'] = store['ccc']


def test_replace_process():
    """replace a process"""
    store = get_toy_store()
    store['process4'] = ToyProcess({'name': 'process4'})
    # store['process4']['A'] = Store()  # TODO: this should give an error

    # replace a process with different ports entirely
    store['process1'] = Qo({})

    # replace a process with a subset of the same ports


def test_connect_to_new_store():
    """
    topology before: 
    process3: {
        'A': ('aaa',),
        'B': ('bbb',),}
        
    topology after:
    process3: {
        'A': ('ddd',),
        'B': ('bbb',),}
    """
    store = get_toy_store()

    # connect port A to a new store ddd
    store['ddd'] = Store({})
    store['process3']['A'] = store['ddd']


def test_set_value():
    store = get_toy_store()
    # set a value through a port
    store['process1']['A']['a'] = 2


def test_run_store_in_experiment():
    """put a store in an experiment and run it"""
    store = get_toy_store()
    experiment = Experiment(store)
    experiment.update(10)



if __name__ == '__main__':
    test_insert_process()
    test_rewire_ports()
    # test_replace_process()
    # test_connect_to_new_store()
