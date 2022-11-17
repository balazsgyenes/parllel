from operator import setitem
import pytest

import multiprocessing as mp
import numpy as np

from parllel.arrays.sharedmemory import SharedMemoryArray, RotatingSharedMemoryArray
from parllel.arrays.managedmemory import ManagedMemoryArray, RotatingManagedMemoryArray


@pytest.fixture(params=["fork", "spawn"], scope="module")
def mp_ctx(request):
    return mp.get_context(request.param)

@pytest.fixture(params=[
    SharedMemoryArray, RotatingSharedMemoryArray,
    ManagedMemoryArray, RotatingManagedMemoryArray
    ], scope="module")
def ArrayClass(request):
    if issubclass(request.param, ManagedMemoryArray):
        pytest.xfail("Currently broken: 'BufferError: cannot close exported pointers exist'")
    return request.param

@pytest.fixture(scope="module")
def shape():
    return (4, 4, 4)

@pytest.fixture(params=[np.float32], scope="module")
def dtype(request):
    return request.param

@pytest.fixture
def blank_array(ArrayClass, shape, dtype):
    array = ArrayClass(shape=shape, dtype=dtype)
    yield array
    array.close()
    array.destroy()

@pytest.fixture
def np_array(shape, dtype):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

@pytest.fixture
def array(blank_array, np_array):
    blank_array[:] = np_array
    return blank_array


def get_array_shape(pipe, array):
    pipe.send(array.shape)


class TestSharedMemoryArray:
    def test_setitem_single(self, array, np_array, mp_ctx):
        location = (0, 1, 2)
        value = -7

        p = mp_ctx.Process(target=setitem, args=(array, location, value))
        p.start()
        p.join()

        np_array[location] = value
        assert np.array_equal(array, np_array)

    def test_setitem_slice(self, array, np_array, mp_ctx):
        location = (3, slice(1,2))
        value = -7

        p = mp_ctx.Process(target=setitem, args=(array, location, value))
        p.start()
        p.join()

        np_array[location] = value
        assert np.array_equal(array, np_array)

    def test_setitem_subarray(self, array, np_array, mp_ctx):
        subarray = array[2, :2]
        subarray = subarray[1:, :]

        location = (0, 3)
        value = -7

        p = mp_ctx.Process(target=setitem, args=(subarray, location, value))
        p.start()
        p.join()

        np_subarray = np_array[2, :2]
        np_subarray = np_subarray[1:, :]
        np_subarray[location] = value
        assert np.array_equal(array, np_array)

    def test_subarray_shape(self, array, np_array, mp_ctx):
        subarray = array[:3, 2]
        subarray = subarray[:, 2:]

        parent_pipe, child_pipe = mp_ctx.Pipe()
        p = mp_ctx.Process(target=get_array_shape, args=(child_pipe, subarray))
        p.start()
        subarray_shape = parent_pipe.recv()
        p.join()

        np_subarray = np_array[:3, 2]
        np_subarray = np_subarray[:, 2:]
        assert subarray_shape == np_subarray.shape
