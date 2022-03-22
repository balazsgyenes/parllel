import functools

import pytest
import numpy as np

from parllel.arrays.array import Array


@pytest.fixture
def shape():
    return (4, 4, 4, 4)

@pytest.fixture
def array_clones(shape):
    array = Array(shape=shape, dtype=np.float32)
    np_array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    array[:] = np_array
    return array, np_array


class TestArray:
    def test_no_args(self):
        with pytest.raises(TypeError):
            _ = Array()

    def test_wrong_dtype(self):
        with pytest.raises(AssertionError):
            shape = (4, 4, 4)
            _ = Array(shape=shape, dtype=list)

    def test_zeroes(self):
        shape = (4, 4, 4)
        array = Array(shape=shape, dtype=np.float32)
        assert np.array_equal(array, np.zeros(shape, dtype=np.float32))

    def test_dtype(self):
        array = Array(shape=(4, 4, 4), dtype=np.float32)
        array[0, 0, 0] = 5
        assert np.asarray(array).dtype == np.float32
        assert np.asarray(array[0, 0, 0]).dtype == np.float32
    
    def test_setitem_single(self, array_clones):
        array, np_array = array_clones
        array[1, 2, 3] = 5.5
        np_array[1, 2, 3] = 5.5
        assert np.array_equal(array, np_array)

    def test_setitem_slices(self, array_clones):
        array, np_array = array_clones
        array[0, 2:3, :2] = 6.5
        np_array[0, 2:3, :2] = 6.5
        assert np.array_equal(array, np_array)

    def test_setitem_ellipsis(self, array_clones):
        array, np_array = array_clones
        array[..., 0] = 7.5
        np_array[..., 0] = 7.5
        assert np.array_equal(array, np_array)

    def test_setitem_reversed_slice(self, array_clones):
        array, np_array = array_clones
        values = np.arange(4, dtype=np.float32) + 0.5
        array[::-1, 0, 0] = values
        np_array[::-1, 0, 0] = values
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_single(self, array_clones):
        array, np_array = array_clones
        subarray, np_subarray = array[1:2, 0], np_array[1:2, 0]
        subarray[0] = 9.5
        np_subarray[0] = 9.5
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_slices(self, array_clones):
        array, np_array = array_clones
        subarray, np_subarray = array[1:2, 0], np_array[1:2, 0]
        subarray[2:3] = 8.5
        np_subarray[2:3] = 8.5
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_ellipsis(self, array_clones):
        array, np_array = array_clones
        subarray, np_subarray = array[1:2, 0], np_array[1:2, 0]
        subarray[...] = 7.5
        np_subarray[...] = 7.5
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_reversed_slice(self, array_clones):
        array, np_array = array_clones
        subarray, np_subarray = array[1:3, 0], np_array[1:3, 0]

        shape = (4, 4)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        values = values + 0.5  # increment so the values are unique

        subarray[1, ::-1] = values
        np_subarray[1, ::-1] = values
        assert np.array_equal(array, np_array)

    def test_element_setitem(self, array_clones):
        array, np_array = array_clones
        element = array[0, 1, 2, 3]

        element[:] = 0.5
        np_array[0, 1, 2, 3] = 0.5  # ndarray does not support item assignment
        assert np.array_equal(array, np_array)

    def test_getitem(self, array_clones):
        array, np_array = array_clones
    
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

    def test_getitem_consecutively(self, array_clones):
        array, np_array = array_clones

        array = array[1:3]
        array = array[0, 0:3]
        array = array[:]
        array = array[:, ::-1]
        assert array.shape == (3, 4, 4)
        assert np.asarray(array).shape == (3, 4, 4)
        assert np.array_equal(array, np_array[1, 0:3, ::-1])

    def test_getitem_single_element(self, array_clones):
        array, np_array = array_clones
        element = array[0, 1, 2, 3]
        np_element = np_array[0, 1, 2, 3]

        assert np.array_equal(element, np_element)
        assert np.asarray(element) == np_element
        assert np.asarray(element) == float(np_element)

    def test_indexhistory(self, array_clones):
        array, _ = array_clones

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

    def test_array_reconstruction(self, array_clones, shape):
        array, np_array = array_clones

        array = array[3]
        array = array[:]
        array = array[0:2]
        array = array[:, -2:]

        array2 = Array(shape=shape, dtype=np.float32)
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
