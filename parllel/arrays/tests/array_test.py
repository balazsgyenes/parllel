import functools
from operator import getitem

import pytest
import numpy as np

from parllel.arrays.array import Array
from parllel.arrays.indices import add_indices, compute_indices, predict_copy_on_index, shape_from_indices
from parllel.arrays.managedmemory import ManagedMemoryArray, RotatingManagedMemoryArray
from parllel.arrays.rotating import RotatingArray
from parllel.arrays.sharedmemory import RotatingSharedMemoryArray, SharedMemoryArray


@pytest.fixture(params=[
    Array, RotatingArray,
    SharedMemoryArray, RotatingSharedMemoryArray,
    ManagedMemoryArray, RotatingManagedMemoryArray,
    ], scope="module")
def ArrayClass(request):
    if issubclass(request.param, ManagedMemoryArray):
        pytest.xfail("Currently broken: 'BufferError: cannot close exported pointers exist'")
    return request.param

@pytest.fixture(scope="module")
def shape():
    return (4, 4, 4)

@pytest.fixture(params=[np.float32, np.int32], scope="module")
def dtype(request):
    return request.param

@pytest.fixture(params=[1, 2], ids=["padding=1", "padding=2"], scope="module")
def padding(request):
    return request.param

@pytest.fixture
def blank_array(ArrayClass, shape, dtype, padding):
    if issubclass(ArrayClass, RotatingArray):
        array = ArrayClass(shape=shape, dtype=dtype, padding=padding)
    else:
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


class TestArray:
    def test_no_args(self, ArrayClass):
        with pytest.raises(TypeError):
            _ = ArrayClass()

    def test_wrong_dtype(self, ArrayClass, shape):
        with pytest.raises(TypeError):
            _ = ArrayClass(shape=shape, dtype=list)

    def test_init(self, blank_array, shape, dtype):
        assert np.array_equal(blank_array, np.zeros(shape, dtype))
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
        values = np.arange(4, dtype=dtype) * 2  # scale so the values are unique
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
        subarray[2:3] = -7
        np_subarray[2:3] = -7
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
        element[:] = -7
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

    def test_getitem(self, array, np_array, shape):
        assert array.shape == shape
        assert np.asarray(array).shape == shape
        assert np.array_equal(array, np_array)

        array = array[:]
        assert array.shape == shape
        assert np.asarray(array).shape == shape
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
        array = array[1:3]
        array = array[0, 0:3]
        array = array[:]
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

    def test_array_reconstruction(self, array):
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

class TestComputeIndices:
    @pytest.mark.parametrize("indices", [
        pytest.param(
            (slice(None, None, 2), 2, slice(1)),
            id="slices"
        ),
        pytest.param(
            (slice(2, None, -1), slice(2), slice(2, None)),
            id="negative step"
        ),
        pytest.param(
            (slice(3), slice(None, None, 2), slice(2, None)),
            id="step > 1"
        ),
        pytest.param(
            (0, 1, 2),
            id="element",
            marks=pytest.mark.xfail(reason="Indexing a scalar results in a copy")
        ),
    ])
    def test_single_index_op(self, np_array, indices):
        np_subarray = np_array[indices]
        indices = compute_indices(np_array, np_subarray)
        assert np.array_equal(np_subarray, np_array[indices])

class TestPredictCopyOnIndex:
    def test_predict_copy_on_index(self):
        assert predict_copy_on_index((2,3,4),(0,0,0))
        assert not predict_copy_on_index((2,3,4), (0,1))
        assert not predict_copy_on_index((9,3,2), (slice(4,5),0,0))

class TestAddIndices:
    @pytest.mark.parametrize("index_history", [
        pytest.param([
            slice(1, 5),
            (2, slice(None, 3)),
            slice(1, 2),
            ], id="slices"
        ),
        pytest.param([
            (Ellipsis, 2),
            (slice(None, 7), slice(1, -1)),
            (3, slice(1, -1)),
            ], id="Ellipsis"
        ),
        pytest.param([
            (2, slice(1, -1)),
            (slice(None), 3),
            1,
            ], id="element"
        ),
        pytest.param([
            slice(None, 1, -1),
            1,
            (slice(None, 2), 3),
            ], id="negative step"
        ),
        pytest.param([
            slice(2, None, -1),
            ], id="negative step ending at None"
        ),
    ])
    def test_index_history(self, np_array, index_history):
        base_shape = np_array.shape
        subarray = np_array
        curr_indices = [slice(None) for _ in base_shape]

        for indices in index_history:
            subarray = subarray[indices]
            curr_indices = add_indices(base_shape, curr_indices, indices)

        assert np.array_equal(subarray, np_array[tuple(curr_indices)])

class TestShapeFromIndices:
    @pytest.mark.parametrize("indices", [
        pytest.param(
            (slice(None, None, 2), 2, slice(1)),
            id="slices"
        ),
        pytest.param(
            (slice(2, None, -1), slice(2), slice(2, None)),
            id="negative step ending at None"
        ),
        pytest.param(
            (slice(3), slice(None, None, 2), slice(2, None)),
            id="step > 1"
        ),
        pytest.param(
            (0, 1, 2),
            id="element"
        ),
    ])
    def test_shape(self, np_array, indices):
        base_shape = np_array.shape
        cleaned_indices = tuple(
            slice(*index.indices(size)) if isinstance(index, slice) else index
            for index, size in zip(indices, base_shape)
        )
        shape = shape_from_indices(np_array.shape, cleaned_indices)
        # IMPORTANT: np_array must be indexed with original indices and not the
        # cleaned indices, because these give different results for
        # slice(2, None, -1) -> slice(2, -1, -1)
        assert shape == np_array[indices].shape
