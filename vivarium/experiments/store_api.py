import argparse
import pytest

from vivarium.composites.toys import Qo, ToyProcess, ToyComposer
from vivarium.core.process import Process
from vivarium.core.engine import Engine
from vivarium.core.store import Store


def get_toy_store() -> Store:
    """get a store to test the api with"""
    composer = ToyComposer({})
    composite = composer.generate()
    store = composite.generate_store()
    return store


def test_insert_process() -> Store:
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
                'port1': {'_path': ('store_C',), 'var_a': ('var_d',)},
                'port2': ('store_A',)}},
        'initial_state': {}})

    assert isinstance(
        store['process3'].value, Process), \
        'process3 not inserted successfully'
    return store


def test_rewire_ports() -> None:
    """connect a process' ports to different store"""
    store = test_insert_process()

    # connect process1's port1 to the store at process3's port1
    store['process1']['port1'] = store['process3']['port1']
    assert store['process1']['port1'] == store['process3']['port1']

    # this should give the same result
    store = test_insert_process()
    store['process1', 'port1'] = store['process3', 'port1']
    assert store['process1']['port1'] == store['process3']['port1']

    # connect process2's port2 to store store_A
    store = test_insert_process()
    store['process2', 'port2'] = store['store_A']
    assert store['process2', 'port2', 'var_a'] == store['store_A', 'var_a']

    # turn variable 'var_a' into 'var_b'
    store = test_insert_process()
    store['process2', 'port2', 'var_a'] = store['store_A', 'var_b']
    assert store['process2', 'port2', 'var_a'] == store['store_A', 'var_b']


def test_embedded_rewire_ports() -> None:
    """rewire process ports embedded down in the hierarchy"""
    composer = ToyComposer({})

    # embed further down a path
    composite = composer.generate(path=('down1', 'down2'))
    store = composite.generate_store()

    # assert process2 is still connected to store_C
    assert store['down1', 'down2', 'process2', 'port2'] == \
           store['down1', 'down2', 'store_C']

    # rewire process2 port2 to store_A, and assert change of wiring
    store['down1', 'down2', 'process2', 'port2'] = \
        store['down1', 'down2', 'store_A']
    assert store['down1', 'down2', 'process2', 'port2', 'var_a'] == \
           store['down1', 'down2', 'store_A', 'var_a']


def test_replace_process() -> None:
    """replace a process"""
    store = get_toy_store()
    process4 = ToyProcess()  # {'name': 'process4'}
    store['process4'] = process4
    assert store['process4'].value == process4

    # replace a process with different ports entirely
    store['process1'] = Qo({})

    # test if initial values are kept the same, and are not overwritten
    store['A', 'a'] = 11
    store['process1'] = ToyProcess({})
    assert store['A', 'a'].value == 11


def test_disconnected_store_failure() -> None:
    """Test that inserting a Store into the tree results in an exception"""
    store = get_toy_store()

    with pytest.raises(Exception):
        store['ddd'] = Store({})
        store['process1', 'port1'] = store['store_D']

    with pytest.raises(Exception):
        store['process1', 'port1'] = Store({'_value': 'NEW STORE'})


# def test_connect_to_new_store() -> None:
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
#     assert store['ddd']['a'] == 0


def test_set_value() -> None:
    """set a value through a port"""
    store = get_toy_store()
    store['process1']['port1']['var_a'] = 5
    assert store['process1']['port1']['var_a'].value == 5


def test_run_store_in_experiment() -> None:
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
    parser.add_argument(
        '--name', '-n', default=[], nargs='+', help='test ids to run')
    args = parser.parse_args()
    run_all = not args.name

    for name in args.name:
        test_library[name]()
    if run_all:
        for name, test in test_library.items():
            test()
