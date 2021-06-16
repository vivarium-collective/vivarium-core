import argparse
import pytest

from vivarium.composites.toys import Qo, ToyProcess, ToyComposer
from vivarium.core.store import Store
from vivarium.core.process import Process
from vivarium.core.engine import Engine
from vivarium.core.store import Store, topology_path


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
        'processes': {'process3': process},
        'topology': {
            'process3': {
                'A': {'_path': ('ccc',), 'a': ('d',)},
                'B': ('aaa',)}},
        'initial_state': {}})

    path, remaining = topology_path(store.get_topology(), ('process1', 'A', 'b'))

    assert isinstance(store['process3'].value, Process), 'process3 not inserted successfully'
    return store


def test_rewire_ports():
    """connect a process' ports to different store"""
    store = test_insert_process()

    # connect process1's port A to the store at process3's port A
    store['process1']['A'] = store['process3']['A']
    assert store['process1']['A'] == store['process3']['A']

    # this should give the same result
    store = test_insert_process()
    store['process1', 'A'] = store['process3', 'A']
    assert store['process1']['A'] == store['process3']['A']

    # connect process2's port B to store aaa
    store = test_insert_process()
    store['process2', 'B'] = store['aaa']
    assert store['process2', 'B', 'a'] == store['aaa', 'a']

    # turn variable 'a' into 'd'
    store = test_insert_process()
    store['process2', 'B', 'a'] = store['aaa', 'b']
    assert store['process2', 'B', 'a'] == store['aaa', 'b']


def test_embedded_rewire_ports():
    """rewire process ports embedded down in the hierarchy"""
    composer = ToyComposer({
        'A':  {'name': 'process1'},
        'B': {'name': 'process2'}})

    # embed further down a path
    composite = composer.generate(path=('down1', 'down2'))
    store = composite.generate_store()

    # assert process2 is still connected to ccc
    assert store['down1', 'down2', 'process2', 'B'] == store['down1', 'down2', 'ccc']

    # rewire process2 port B to aaa, and assert change of wiring
    store['down1', 'down2', 'process2', 'B'] = store['down1', 'down2', 'aaa']
    assert store['down1', 'down2', 'process2', 'B', 'a'] == store['down1', 'down2', 'aaa', 'a']


def test_replace_process():
    """replace a process"""
    store = get_toy_store()
    store['process4'] = ToyProcess({'name': 'process4'})

    # replace a process with different ports entirely
    store['process1'] = Qo({})

    # test if initial values are kept the same, and are not overwritten
    store['A', 'a'] = 11
    store['process1'] = ToyProcess({})
    assert store['A', 'a'].value == 11


def test_disconnected_store_failure():
    """Test that inserting a Store into the tree results in an exception"""
    store = get_toy_store()

    with pytest.raises(Exception):
        store['ddd'] = Store({})
        store['process1', 'A'] = store['ddd']

    with pytest.raises(Exception):
        store['process1', 'A'] = Store({'_value': 'NEW STORE'})


# def test_connect_to_new_store():
#     """
#     topology before:
#     process3: {
#         'A': ('aaa',),
#         'B': ('bbb',),}
#
#     topology after:
#     process3: {
#         'A': ('ddd',),
#         'B': ('bbb',),}
#     """
#     store = get_toy_store()
#
#     # connect a new store to the tree
#     store['ddd'] = Store({})
#
#     # connect port A to the new store ddd
#     store['process2']['A'] = Store({})
#
#     assert store['process2'].topology == {'A': ('ddd',), 'B': ('ccc',)}
#     assert store['ddd']['a'] == 0, "store 'ddd' variable 'a' is not being initialized to the default value of 0"


def test_set_value():
    """set a value through a port"""
    store = get_toy_store()
    store['process1']['A']['a'] = 5
    assert store['process1']['A']['a'].value == 5


def test_run_store_in_experiment():
    """put a store in an experiment and run it"""
    store = get_toy_store()
    experiment = Engine({'store': store})
    experiment.update(10)
    data = experiment.emitter.get_data()

    assert experiment.processes['process1'] == store['process1'].value
    assert experiment.processes['process2'] == store['process2'].value
    assert data[10.0] != data[0.0]

    print(data)


test_library = {
    '1': test_insert_process,
    '2': test_rewire_ports,
    '3': test_embedded_rewire_ports,
    '4': test_replace_process,
    '5': test_disconnected_store_failure,
    # '6': test_connect_to_new_store,
    '7': test_set_value,
    '8': test_run_store_in_experiment,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='store API')
    parser.add_argument('--name', '-n', default=[], nargs='+', help='test ids to run')
    args = parser.parse_args()
    run_all = not args.name

    for name in args.name:
        test_library[name]()
    if run_all:
        for name, test in test_library.items():
            test()
