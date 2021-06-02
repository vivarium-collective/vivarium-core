import argparse
import pytest

from vivarium.composites.toys import Qo, ToyProcess, ToyComposer
from vivarium.core.store import Store
from vivarium.core.process import Process
from vivarium.core.experiment import Experiment
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

    # assert store['process3']['A'].path_for() ==

    # this should give the same result
    store = test_insert_process()
    store['process1', 'A'] = store['process3', 'A']
    assert store['process1']['A'] == store['process3']['A']

    # connect process2's port B to store aaa
    store = test_insert_process()
    store['process2']['B'] = store['aaa']
    assert store['process2', 'B', 'a'] == store['aaa', 'a']

    # turn variable 'a' into 'd'
    store = test_insert_process()
    store['process2']['B', 'a'] = store['aaa', 'b']
    assert store['process2', 'B', 'a'] == store['aaa', 'b']

    import ipdb;
    ipdb.set_trace()


def test_replace_process():
    """replace a process"""
    store = get_toy_store()
    store['process4'] = ToyProcess({'name': 'process4'})

    # replace a process with different ports entirely
    store['process1'] = Qo({})

    # test if initial values are kept the same, and are not overwritten
    store['A', 'a'] = 1
    store['process1'] = ToyProcess({})
    assert store['A', 'a'].value == 1


def test_disconnected_store_failure():
    store = get_toy_store()
    with pytest.raises(Exception):
        store['process1']['A'] = Store()  # TODO: this should give an error


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
#     store.inspect()
#
#     # connect port A to the new store ddd
#     store['process3']['A'] = store['ddd']


def test_set_value():
    """set a value through a port"""
    store = get_toy_store()
    store['process1']['A']['a'] = 5
    assert store['process1']['A']['a'].value == 5


def test_run_store_in_experiment():
    """put a store in an experiment and run it"""
    store = get_toy_store()
    experiment = Experiment(store)
    experiment.update(10)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='store API')
    parser.add_argument('--number', '-n', default=[], nargs='+', help='test ids to run')
    args = parser.parse_args()
    run_all = not args.number

    if '1' in args.number or run_all:
        test_insert_process()
    if '2' in args.number or run_all:
        test_rewire_ports()
    if '3' in args.number or run_all:
        test_replace_process()
    # if '4' in args.number or run_all:
    #     test_connect_to_new_store()
    if '5' in args.number or run_all:
        test_set_value()
    if '6' in args.number or run_all:
        test_run_store_in_experiment()

