import numpy as np
import pytest

from parllel.arrays.indices import add_indices, compute_indices, predict_copy_on_index, shape_from_indices


@pytest.fixture(scope="module")
def shape():
    return (4, 4, 4)

@pytest.fixture(params=[np.float32, np.int32], scope="module")
def dtype(request):
    return request.param

@pytest.fixture
def np_array(shape, dtype):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)


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
            ], id="slices",
        ),
        pytest.param([
            (Ellipsis, 2),
            (slice(None, 7), slice(1, -1)),
            (3, slice(1, -1)),
            ], id="Ellipsis",
        ),
        pytest.param([
            (2, slice(1, -1)),
            (slice(None), 3),
            1,
            ], id="element",
        ),
        pytest.param([
            slice(None, 1, -1),
            1,
            (slice(None, 2), 3),
            ], id="negative step",
        ),
        pytest.param([
            slice(2, None, -1),
            ], id="negative step ending at None",
            marks=pytest.mark.skip,
        ),
    ])
    def test_add_indices(self, np_array, index_history):
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
    def test_shape_from_indices(self, np_array, indices):
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
