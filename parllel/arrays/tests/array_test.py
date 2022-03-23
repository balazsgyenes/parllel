import functools

import pytest
import numpy as np

from parllel.arrays.array import Array


@pytest.fixture
def shape():
    return (4, 4, 4, 4)

@pytest.fixture
def dtype():
    return np.float32

@pytest.fixture
def blank_array(shape, dtype):
    return Array(shape=shape, dtype=dtype)

@pytest.fixture
def np_array(shape, dtype):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

@pytest.fixture
def array(blank_array, np_array):
    blank_array[:] = np_array
    return blank_array


class TestArray:
    def test_no_args(self):
        with pytest.raises(TypeError):
            _ = Array()

    def test_wrong_dtype(self):
        with pytest.raises(AssertionError):
            shape = (4, 4, 4)
            _ = Array(shape=shape, dtype=list)

    def test_init(self, blank_array, shape, dtype):
        assert np.array_equal(blank_array, np.zeros(shape, dtype))
        assert blank_array.dtype == dtype
        assert np.asarray(blank_array).dtype == dtype

    def test_dtype(self, blank_array):
        blank_array[0, 0, 0] = 5
        assert blank_array.dtype == np.float32
        assert np.asarray(blank_array).dtype == np.float32
        assert blank_array[0, 0, 0].dtype == np.float32
        assert np.asarray(blank_array[0, 0, 0]).dtype == np.float32
    
    def test_setitem_single(self, array, np_array):
        array[1, 2, 3] = 5.5
        np_array[1, 2, 3] = 5.5
        assert np.array_equal(array, np_array)

    def test_setitem_slices(self, array, np_array):
        array[0, 2:3, :2] = 6.5
        np_array[0, 2:3, :2] = 6.5
        assert np.array_equal(array, np_array)

    def test_setitem_ellipsis(self, array, np_array):
        array[..., 0] = 7.5
        np_array[..., 0] = 7.5
        assert np.array_equal(array, np_array)

    def test_setitem_reversed_slice(self, array, np_array, dtype):
        values = np.arange(4, dtype=dtype) + 0.5
        array[::-1, 0, 0] = values
        np_array[::-1, 0, 0] = values
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_single(self, array, np_array):
        subarray, np_subarray = array[1:2, 0], np_array[1:2, 0]
        subarray[0] = 9.5
        np_subarray[0] = 9.5
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_slices(self, array, np_array):
        subarray, np_subarray = array[1:2, 0], np_array[1:2, 0]
        subarray[2:3] = 8.5
        np_subarray[2:3] = 8.5
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_ellipsis(self, array, np_array):
        subarray, np_subarray = array[1:2, 0], np_array[1:2, 0]
        subarray[...] = 7.5
        np_subarray[...] = 7.5
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_reversed_slice(self, array, np_array):
        subarray, np_subarray = array[1:3, 0], np_array[1:3, 0]

        shape = (4, 4)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        values = values + 0.5  # increment so the values are unique

        subarray[1, ::-1] = values
        np_subarray[1, ::-1] = values
        assert np.array_equal(array, np_array)

    def test_element_setitem(self, array, np_array):
        element = array[0, 1, 2, 3]
        element[:] = 0.5
        np_array[0, 1, 2, 3] = 0.5  # ndarray does not support item assignment
        assert np.array_equal(array, np_array)

        element[...] = -7
        np_array[0, 1, 2, 3] = -7  # ndarray does not support item assignment
        assert np.array_equal(array, np_array)

    def test_element_setitem_slice(self, array):
        element = array[0, 1, 2, 3]
        with pytest.raises(IndexError):
            element[0] = 0.5

        with pytest.raises(IndexError):
            element[1:] = 0.5

    def test_getitem(self, array, np_array):
        assert array.shape == (4, 4, 4, 4)
        assert np.asarray(array).shape == (4, 4, 4, 4)
        assert np.array_equal(array, np_array)

        array = array[:]
        assert array.shape == (4, 4, 4, 4)
        assert np.asarray(array).shape == (4, 4, 4, 4)
        assert np.array_equal(array, np_array)

        array = array[2]
        assert array.shape == (4, 4, 4)
        assert np.asarray(array).shape == (4, 4, 4)
        assert np.array_equal(array, np_array[2])

        array = array[1:3]
        assert array.shape == (2, 4, 4)
        assert np.asarray(array).shape == (2, 4, 4)
        assert np.array_equal(array, np_array[2, 1:3])

        array = array[:, ::2]
        assert array.shape == (2, 2, 4)
        assert np.asarray(array).shape == (2, 2, 4)
        assert np.array_equal(array, np_array[2, 1:3, ::2])

    def test_getitem_consecutively(self, array, np_array):
        array = array[1:3]
        array = array[0, 0:3]
        array = array[:]
        array = array[:, ::-1]
        assert array.shape == (3, 4, 4)
        assert np.asarray(array).shape == (3, 4, 4)
        assert np.array_equal(array, np_array[1, 0:3, ::-1])

    def test_getitem_single_element(self, array, np_array):
        element = array[0, 1, 2, 3]
        np_element = np_array[0, 1, 2, 3]

        assert np.array_equal(element, np_element)
        assert np.asarray(element) == np_element
        assert np.asarray(element) == float(np_element)

    def test_indexhistory(self, array):
        assert array.index_history == ()

        array = array[:]
        assert array.index_history == (slice(None),)

        array = array[2]
        assert array.index_history == (slice(None), 2)

        array = array[1:3]
        assert array.index_history == (slice(None), 2, slice(1, 3))

        array = array[:, ::2]
        assert array.index_history == (
            slice(None),
            2,
            slice(1, 3),
            (slice(None), slice(None, None, 2)),
        )

    def test_array_reconstruction(self, array, np_array, shape, dtype):
        array = array[3]
        array = array[:]
        array = array[0:2]
        array = array[:, -2:]

        array2 = Array(shape=shape, dtype=dtype)
        array2[:] = np_array
        # apply array's index history to array2
        array2 = functools.reduce(
            lambda buf, index: buf[index],
            array.index_history,
            array2
        )

        assert np.array_equal(array, array2)
        assert all(el1 == el2 for el1, el2
            in zip(array.index_history, array2.index_history))
