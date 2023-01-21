from itertools import islice
from typing import Tuple, Union


import numpy as np
import numpy.random as random
import pytest

from parllel.arrays.indices import (
    add_locations, add_indices, index_slice,
    compute_indices,
    predict_copy_on_index,
    shape_from_indices,
)


PROB_SLICE = 0.5
PROB_INDEX_NEGATIVE = 0.5
PROB_STEP_NEGATIVE = 0.5
MAX_STEP = 1
PROB_SLICE_EL_NONE = 0.2
def random_index(rng: random.Generator, size: int) -> Union[int, slice]:
    if rng.random() > PROB_SLICE:
        # result is integer index
        index = int(rng.integers(size))
        return index if rng.random() > PROB_INDEX_NEGATIVE else -index
    else:
        assert size > 0

        # result is slice
        if rng.random() > PROB_SLICE_EL_NONE:
            step = int(rng.integers(1, high=MAX_STEP, endpoint=True))
            # step = step if rng.random() > PROB_STEP_NEGATIVE else -step
        else:
            step = None

        if step is not None and step < 0:
            # negative step. flip bounds
            start = int(rng.integers(size - int(np.sqrt(size)), size - 1, endpoint=True))
            start = start if rng.random() > PROB_SLICE_EL_NONE else None

            max_stop = start - 1 if start is not None else size - 2
            if max_stop < 0:
                # best we can do is a slice of size 1 if the stop is None
                stop = None
            else:
                stop = int(rng.integers(0, high=max_stop, endpoint=True))
                stop = stop if rng.random() > PROB_SLICE_EL_NONE else None        

        else:
            # ensures that min_start is 0 if size==1
            min_start = int(np.sqrt(size - 1))
            start = int(rng.integers(min_start, endpoint=True))
            start = start if rng.random() > PROB_SLICE_EL_NONE else None

            min_stop = start + 1 if start is not None else 1
            stop = int(rng.integers(min_stop, high=size, endpoint=True))
            stop = stop if rng.random() > PROB_SLICE_EL_NONE else None        
        
        return slice(start, stop, step)

def random_location(rng: random.Generator, shape: Tuple[int, ...],
) -> Tuple[Union[int, slice], ...]:
    return tuple(
        random_index(rng, size)
        for size in islice(
            shape,
            rng.integers(1, high=len(shape), endpoint=True),
        )
    )


@pytest.fixture(scope="module")
def shape():
    return (20, 20, 20, 20)

@pytest.fixture
def np_array(shape):
    return np.arange(np.prod(shape)).reshape(shape)

@pytest.fixture(scope="module")
def rng():
    return random.default_rng()


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
        ),
    ])
    def test_explicit_cases(self, np_array, index_history):
        subarray = np_array
        curr_indices = [slice(None) for _ in np_array.shape]

        for indices in index_history:
            subarray = subarray[indices]
            curr_indices = add_locations(curr_indices, indices)

        assert np.array_equal(subarray, np_array[tuple(curr_indices)])

    def test_random_cases(self, np_array: np.ndarray, rng: random.Generator):
        for _ in range(500):
            loc1 = random_location(rng, np_array.shape)
            subarray1 = np_array[loc1]
            if not subarray1.shape:
                continue  # if we have indexed a single element already, skip
            loc2 = random_location(rng, subarray1.shape)
            subarray2 = subarray1[loc2]

            loc = add_locations([], loc1)
            loc = add_locations(loc, loc2)
            
            assert np.array_equal(subarray2, np_array[tuple(loc)])


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


class TestShapeFromIndices:
    @pytest.mark.parametrize("indices", [
        pytest.param(
            (slice(None, None, 2), 2, slice(3), slice(10)),
            id="slices"
        ),
        pytest.param(
            (slice(2, None, -1), slice(2), slice(2, None), 5),
            id="negative step ending at None"
        ),
        pytest.param(
            (slice(3), slice(None, None, 2), slice(2, None), -1),
            id="step > 1"
        ),
        pytest.param(
            (0, 1, 2, 3),
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
