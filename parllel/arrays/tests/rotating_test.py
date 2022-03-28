import functools

import pytest
import numpy as np

from parllel.arrays.managedmemory import RotatingManagedMemoryArray
from parllel.arrays.rotating import RotatingArray, shift_index
from parllel.arrays.sharedmemory import RotatingSharedMemoryArray


@pytest.fixture(params=[
    RotatingArray,
    RotatingSharedMemoryArray,
    RotatingManagedMemoryArray,
    ], scope="module")
def ArrayClass(request):
    return request.param

@pytest.fixture(scope="module")
def shape():
    return (10, 4, 4)

@pytest.fixture(params=[np.float32, np.int32], scope="module")
def dtype(request):
    return request.param

@pytest.fixture(params=[1, 2], ids=["padding=1", "padding=2"], scope="module")
def padding(request):
    return request.param

@pytest.fixture
def blank_array(ArrayClass, shape, dtype, padding):
    return ArrayClass(shape=shape, dtype=dtype, padding=padding)

@pytest.fixture
def np_array(shape, dtype):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

@pytest.fixture
def previous_region(np_array, padding):
    # flip data in trailing dims
    return np_array[:padding, ::-1, ::-1].copy()

@pytest.fixture
def next_region(np_array, padding):
    # flip data in trailing dims, scale to make unique
    return np_array[:padding, ::-1, ::-1] * 2

@pytest.fixture
def array(blank_array, np_array, padding, previous_region, next_region):
    blank_array[:] = np_array
    blank_array[-padding:0] = previous_region
    last = blank_array.last
    blank_array[(last + 1):(last + padding + 1)] = next_region
    return blank_array


class TestRotatingArray:
    def test_negative_padding(self, ArrayClass, shape, dtype, padding):
        with pytest.raises(ValueError):
            _ = ArrayClass(shape=shape, dtype=dtype, padding=-padding)

    def test_init(self, blank_array, shape, dtype):
        assert blank_array.shape == shape
        assert np.asarray(blank_array).shape == shape
        assert blank_array.dtype == dtype
        assert np.asarray(blank_array).dtype == dtype

    def test_setitem_single(self, array, np_array, next_region):
        array[array.first - 1] = -7
        array[array.last + 1] = -8
        ones = np.ones_like(next_region[0])

        assert np.array_equal(array, np_array)
        assert np.array_equal(array[array.first - 1], ones * -7)
        np.array_equal(array[array.last + 1], ones * -8)

    def test_setitem_slices(self, array, shape, dtype, np_array):
        array[array.first - 1 : 1] = -7
        ones = np.ones((2,) + shape[1:], dtype)
        assert np.array_equal(array[array.first - 1 : 1], ones * -7)
        assert np.array_equal(array[1:], np_array[1:])

    def test_setitem_all(self, array, padding, previous_region, next_region):
        array[:] = -7
        first = array.first
        last = array.last
        assert np.array_equal(array[first - padding : 0], previous_region)
        assert np.array_equal(array[last + 1 : last + padding + 1], next_region)

    def test_setitem_ellipsis(self, array, padding, previous_region, next_region):
        array[..., 1] = -7
        first = array.first
        last = array.last
        assert np.array_equal(array[first - padding : 0], previous_region)
        assert np.array_equal(array[last + 1 : last + padding + 1], next_region)

    def test_setitem_beyond_last(self, blank_array, padding):
        with pytest.raises(IndexError):
            blank_array[blank_array.last + padding + 1] = 1

    def test_setitem_beyond_beginning(self, blank_array, padding):
        with pytest.raises(IndexError):
            blank_array[blank_array.first - padding - 1] = 1

    def test_getitem(self, array, np_array, previous_region):
        array = array[array.first - 1:]
        assert array.shape == (11, 4, 4)
        assert np.asarray(array).shape == (11, 4, 4)
        values = np.concatenate((previous_region[-1:], np_array))
        assert np.array_equal(array, values)

        array = array[2]
        assert array.shape == (4, 4)
        assert np.asarray(array).shape == (4, 4)
        assert np.array_equal(array, np_array[1])

    def test_getitem_consecutively(self, array, np_array):
        array = array[array.first - 1:]
        array = array[2]
        assert array.shape == (4, 4)
        assert np.asarray(array).shape == (4, 4)
        assert np.array_equal(array, np_array[1])

    def test_rotate(self, array, np_array, padding, next_region):
        array.rotate()
        first = array.first
        last = array.last
        # elements just before last have been moved into padding
        assert np.array_equal(array[first - padding:first], array[last - padding + 1:])
        # elements just after last have been moved to the beginning
        assert np.array_equal(array[:padding], next_region)
        # the rest of the array body is unaffected
        assert np.array_equal(array[padding:], np_array[padding:])

    def test_indexhistory(self, array):
        assert array.index_history == ()

        array = array[array.first - 1:]
        assert array.index_history == (slice(-1, None), )

        array = array[1]
        assert array.index_history == (slice(-1, None), 1)

        array = array[1:3]
        assert array.index_history == (slice(-1, None), 1, slice(1, 3))

        array = array[array.first - 1, ::2]
        assert array.index_history == (
            slice(-1, None),
            1,
            slice(1, 3),
            (-1, slice(None, None, 2)),
        )

    def test_array_reconstruction(self, array):
        # same as Array test, just start with negative index here
        subarray1 = array
        subarray1 = subarray1[subarray1.first - 1:]
        subarray1 = subarray1[:]
        subarray1 = subarray1[0:2]
        subarray1 = subarray1[:, -2:]

        # apply subarray1's index history to array again
        subarray2 = functools.reduce(
            lambda buf, index: buf[index],
            subarray1.index_history,
            array
        )

        assert np.array_equal(subarray1, subarray2)
        assert all(el1 == el2 for el1, el2
            in zip(subarray1.index_history, subarray2.index_history))


class TestIndexShifting:
    def test_shift_integer(self):
        assert shift_index(4, 2) == (6,)
        assert shift_index(0, 1) == (1,)
        assert shift_index(-1, 2) == (1,)
        assert shift_index(-2, 2) == (0,)

    def test_shift_index_too_negative(self):
        with pytest.raises(IndexError):
            _ = shift_index(-2, 1)

    def test_shift_slice(self):
        assert shift_index(slice(1, 5, 2), 2) == (slice(3, 7, 2),)
        assert shift_index(slice(None, None, 2), 1) == (slice(1, -1, 2),)
        assert shift_index(slice(None, 5), 1) == (slice(1, 6, None),)
        assert shift_index(slice(None, 5, 1), 1) == (slice(1, 6, 1),)
        assert shift_index(slice(2, None, 2), 2) == (slice(4, -2, 2),)

    def test_shift_reversed_slice(self):
        assert shift_index(slice(1, 4, -1), 2) == (slice(3, 6, -1),)
        assert shift_index(slice(None, None, -1), 1) == (slice(-2, 0, -1),)
        assert shift_index(slice(None, None, -1), 2) == (slice(-3, 1, -1),)
        assert shift_index(slice(None, 3, -1), 1) == (slice(-2, 4, -1),)
        assert shift_index(slice(5, None, -2), 1) == (slice(6, 0, -2),)
        assert shift_index(slice(6, None, -3), 2) == (slice(8, 1, -3),)

    def test_shift_ellipsis(self):
        assert shift_index(..., 1) == (slice(1, -1), Ellipsis)
        assert shift_index(..., 2) == (slice(2, -2), Ellipsis)
