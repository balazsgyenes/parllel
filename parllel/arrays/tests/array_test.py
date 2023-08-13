import functools
from operator import getitem

import numpy as np
import pytest

from parllel.arrays import Array


# fmt: off
class TestArrayCreation:
    def test_no_args(self):
        with pytest.raises(TypeError):
            _ = Array()

    def test_empty_shape(self, dtype):
        with pytest.raises(ValueError):
            _ = Array(batch_shape=(), dtype=dtype)

    def test_wrong_dtype(self, batch_shape):
        with pytest.raises(ValueError):
            _ = Array(batch_shape=batch_shape, dtype=list)

    def test_calling_array(self, batch_shape, dtype, storage, padding):
        array = Array(batch_shape=batch_shape, dtype=dtype, storage=storage, padding=padding)
        assert array.shape == batch_shape
        assert array.dtype == dtype
        assert array.storage == storage
        assert array.padding == padding

    def test_new_array(self):
        template = Array(batch_shape=(10, 4), dtype=np.float32)
        
        array = template.new_array(batch_shape=(20, 4))
        assert array.shape == (20, 4)
        assert array.dtype == np.float32
        assert array.full.shape == (20, 4)

        array = template.new_array(dtype=np.int32)
        assert array.shape == (10, 4)
        assert array.dtype == np.int32

        array = template.new_array(storage="shared")
        assert array.storage == "shared"
        array.close()

        array = template.new_array(padding=1)
        assert array.padding == 1

        array = template.new_array(full_size=20)
        assert array.shape == (10, 4)
        assert array.full.shape == (20, 4)

    def test_new_array_windowed(self):
        template = Array(batch_shape=(10, 4), dtype=np.float32, full_size=20)

        array = template.new_array(batch_shape=(5, 4))
        assert array.shape == (5, 4)
        assert array.dtype == np.float32
        assert array.full.shape == (5, 4)

        array = template.new_array(batch_shape=(5, 4), inherit_full_size=True)
        assert array.shape == (5, 4)
        assert array.full.shape == (20, 4)

        array = template.new_array(dtype=np.int32)
        assert array.shape == (10, 4)
        assert array.dtype == np.int32

        array = template.new_array(storage="shared")
        assert array.storage == "shared"
        array.close()

        array = template.new_array(padding=1)
        assert array.padding == 1

        array = template.new_array(full_size=40)
        assert array.shape == (10, 4)
        assert array.full.shape == (40, 4)


class TestArray:
    def test_init(self, blank_array, batch_shape, dtype):
        assert np.array_equal(blank_array, np.zeros(batch_shape, dtype))
        assert blank_array.dtype == dtype
        assert np.asarray(blank_array).dtype == dtype

    def test_dtype(self, blank_array, dtype):
        blank_array[0, 0, 0] = -1.1
        assert blank_array.dtype == dtype
        assert np.asarray(blank_array).dtype == dtype
        assert blank_array[0, 0, 0].dtype == dtype
        assert np.asarray(blank_array[0, 0, 0]).dtype == dtype
    
    def test_setitem_single(self, array, np_array):
        array[1, 2, 3] = -7
        np_array[1, 2, 3] = -7
        assert np.array_equal(array, np_array)

    def test_setitem_slices(self, array, np_array):
        array[0, 2:3, :2] = -7
        np_array[0, 2:3, :2] = -7
        assert np.array_equal(array, np_array)

    def test_setitem_all(self, array, np_array):
        array[:] = -7
        np_array[:] = -7
        assert np.array_equal(array, np_array)

    def test_setitem_ellipsis(self, array, np_array):
        array[..., 0] = -7
        np_array[..., 0] = -7
        assert np.array_equal(array, np_array)

    def test_setitem_reversed_slice(self, array, np_array, dtype):
        values = np.arange(10, dtype=dtype) * 2  # scale so the values are unique
        array[::-1, 0, 0] = values
        np_array[::-1, 0, 0] = values
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_single(self, array, np_array):
        subarray, np_subarray = array[1:2, 0], np_array[1:2, 0]
        subarray[0] = -7
        np_subarray[0] = -7
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_slices(self, array, np_array):
        subarray, np_subarray = array[1:2, 0], np_array[1:2, 0]
        subarray[:, 2:3] = -7
        np_subarray[:, 2:3] = -7
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_ellipsis(self, array, np_array):
        subarray, np_subarray = array[1:2, 0], np_array[1:2, 0]
        subarray[...] = -7
        np_subarray[...] = -7
        assert np.array_equal(array, np_array)

    def test_subarrays_setitem_reversed_slice(self, array, np_array):
        subarray, np_subarray = array[1:3, 0], np_array[1:3, 0]

        shape = (4,)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        values = values * 2  # scale so the values are unique

        subarray[1, ::-1] = values
        np_subarray[1, ::-1] = values
        assert np.array_equal(array, np_array)

    def test_element_setitem(self, array, np_array):
        element = array[0, 1, 2]
        element[...] = -7
        np_array[0, 1, 2] = -7  # ndarray does not support item assignment
        assert np.array_equal(array, np_array)

        element[...] = -8
        np_array[0, 1, 2] = -8  # ndarray does not support item assignment
        assert np.array_equal(array, np_array)

    def test_element_setitem_index(self, array):
        element = array[0, 1, 2]
        with pytest.raises(IndexError):
            element[0] = -7

    def test_element_setitem_slice(self, array):
        element = array[0, 1, 2]
        with pytest.raises(IndexError):
            element[1:] = -8

    def test_getitem(self, array, np_array, batch_shape):
        assert array.shape == batch_shape
        assert np.asarray(array).shape == batch_shape
        assert np.array_equal(array, np_array)

        array = array[:]
        assert array.shape == batch_shape
        assert np.asarray(array).shape == batch_shape
        assert np.array_equal(array, np_array)

        array = array[2]
        assert array.shape == (4, 4)
        assert np.asarray(array).shape == (4, 4)
        assert np.array_equal(array, np_array[2])

        array = array[1:3]
        assert array.shape == (2, 4)
        assert np.asarray(array).shape == (2, 4)
        assert np.array_equal(array, np_array[2, 1:3])

        array = array[:, ::2]
        assert array.shape == (2, 2)
        assert np.asarray(array).shape == (2, 2)
        assert np.array_equal(array, np_array[2, 1:3, ::2])

    def test_getitem_consecutively(self, array, np_array):
        array = array[:]
        array = array[1:3]
        array = array[0, 0:3]
        array = array[:, ::-1]
        assert array.shape == (3, 4)
        assert np.asarray(array).shape == (3, 4)
        assert np.array_equal(array, np_array[1, 0:3, ::-1])

    def test_getitem_single_element(self, array, np_array):
        element = array[0, 1, 2]
        np_element = np_array[0, 1, 2]

        assert np.array_equal(element, np_element)
        assert np.asarray(element) == np_element
        assert np.asarray(element) == np_element.item()

    @pytest.mark.skip()
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

    @pytest.mark.skip()
    def test_array_reconstruction(self, array):
        # TODO: compare arrays with np_array
        subarray1 = array
        subarray1 = subarray1[3]
        subarray1 = subarray1[:]
        subarray1 = subarray1[0:2]
        subarray1 = subarray1[:, -2:]

        # apply subarray1's index history to array again
        subarray2 = functools.reduce(getitem, subarray1.index_history, array)

        assert np.array_equal(subarray1, subarray2)
        assert all(el1 == el2 for el1, el2
            in zip(subarray1.index_history, subarray2.index_history))
