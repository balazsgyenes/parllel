from itertools import islice
import math


import numpy as np
import numpy.random as random
import pytest

from parllel.buffers.buffer import Indices
from parllel.arrays.indices import (
    add_locations, add_indices, index_slice,
    clean_slice, init_location,
    compute_indices,
    predict_copy_on_index,
    shape_from_indices,
)


PROB_SLICE = 0.5
PROB_INDEX_NEGATIVE = 0.5
PROB_STEP_NEGATIVE = 0.5
MAX_STEP = 3
PROB_SLICE_EL_NONE = 0.2
def random_int(rng: random.Generator, size: int) -> int:
    assert size > 0
    index = int(rng.integers(size))
    return -index if rng.random() < PROB_INDEX_NEGATIVE else index

def random_slice(
    rng: random.Generator,
    size: int,
    max_step: int = MAX_STEP,
    prob_start_stop_negative: float = PROB_INDEX_NEGATIVE,
    prob_step_negative: float = PROB_STEP_NEGATIVE,
) -> slice:
    if rng.random() > PROB_SLICE_EL_NONE:
        step = int(rng.integers(1, high=max_step, endpoint=True))
        step = -step if rng.random() < prob_step_negative else step
    else:
        step = None

    if step is not None and step < 0:
        # negative step. flip bounds
        start = int(rng.integers(size - math.isqrt(size), size - 1, endpoint=True))
        start = None if rng.random() < PROB_SLICE_EL_NONE else start

        max_stop = start - 1 if start is not None else size - 2
        if max_stop < 0:
            # best we can do is a slice of size 1 if the stop is None
            stop = None
        else:
            stop = int(rng.integers(0, high=max_stop, endpoint=True))
            stop = None if rng.random() < PROB_SLICE_EL_NONE else stop        

    else:
        # ensures that min_start is 0 if size==1
        min_start = math.isqrt(size - 1)
        start = int(rng.integers(min_start, endpoint=True))
        start = None if rng.random() < PROB_SLICE_EL_NONE else start

        min_stop = start + 1 if start is not None else 1
        stop = int(rng.integers(min_stop, high=size, endpoint=True))
        stop = None if rng.random() < PROB_SLICE_EL_NONE else stop        
    
    start = start - size if (
        rng.random() < prob_start_stop_negative and
        start is not None
    ) else start
    stop = stop - size if (
        rng.random() < prob_start_stop_negative and 
        stop is not None and 
        stop < size  # if stop==size, cannot be converted to negative index
    ) else stop

    return slice(start, stop, step)

def random_location(
    rng: random.Generator,
    shape: tuple[int, ...],
    max_step: int = MAX_STEP,
    prob_start_stop_negative: float = PROB_INDEX_NEGATIVE,
    prob_step_negative: float = PROB_STEP_NEGATIVE,
) -> Indices:
    return tuple(
        (
            random_slice(rng, size, max_step, prob_step_negative, prob_start_stop_negative)
            if rng.random() < PROB_SLICE else
            random_int(rng, size)
        )
        for size in islice(
            shape,
            rng.integers(1, high=len(shape), endpoint=True),
        )
    )


@pytest.fixture(params=[
    (20, 20, 20, 20)
], scope="module")
def shape(request):
    return request.param

@pytest.fixture
def np_array(shape):
    return np.arange(np.prod(shape)).reshape(shape)

@pytest.fixture
def vector(shape):
    return np.arange(shape[0]).reshape(shape[:1])

@pytest.fixture(scope="module")
def rng():
    return random.default_rng()

@pytest.fixture(
    params=[0.0, 0.3],
    ids=["pos_point", "pos/neg_point"],
    scope="module",
)
def prob_start_stop_negative(request):
    return request.param

@pytest.fixture(
    params=[0.0, 0.5],
    ids=["pos_step", "pos/neg_step"],
    scope="module",
)
def prob_step_negative(request):
    return request.param

@pytest.fixture(
    params=[1, 3],
    ids=["max_step=1", "max_step=3"],
    scope="module",
)
def max_step(request):
    return request.param


class TestAddIndices:
    @pytest.mark.parametrize("index_history", [
        pytest.param([
            slice(1, 5),
            (2, slice(None, 3)),
            slice(1, 2),
            ], id="slices",
        ),
        pytest.param([
            (3, slice(1, -1)),
            (Ellipsis, 2),
            (slice(None, 7), slice(1, -1)),
            ], id="Ellipsis with after indexing with int",
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
    def test_add_locations_named_cases(self,
        np_array: np.ndarray,
        index_history: list[Indices],
    ):
        subarray = np_array
        curr_indices = init_location(np_array.shape)

        for indices in index_history:
            subarray = subarray[indices]
            curr_indices = add_locations(curr_indices, indices, np_array.shape)

        assert np.array_equal(subarray, np_array[tuple(curr_indices)])

    def test_add_locations(self,
        np_array: np.ndarray,
        rng: random.Generator,
        max_step: int,
        prob_step_negative: float,
        prob_start_stop_negative: float,
    ):
        for _ in range(1000):
            loc1 = random_location(rng, np_array.shape, max_step, prob_step_negative, prob_start_stop_negative)
            subarray1 = np_array[loc1]
            loc1_cleaned = add_locations(init_location(np_array.shape), loc1, np_array.shape)  # clean location

            assert np.array_equal(subarray1, np_array[tuple(loc1_cleaned)])

            if not subarray1.shape:
                continue  # if we have indexed a single element already, skip
            
            loc2 = random_location(rng, subarray1.shape, max_step, prob_step_negative, prob_start_stop_negative)
            subarray2 = subarray1[loc2]
            joined_loc = add_locations(loc1_cleaned, loc2, np_array.shape)
            
            assert np.array_equal(subarray2, np_array[tuple(joined_loc)])

    def test_index_slice(self,
        vector: np.ndarray,
        shape: tuple[int, ...],
        rng: random.Generator,
        max_step: int,
        prob_step_negative: float,
        prob_start_stop_negative: float,
    ):
        for _ in range(1000):
            loc1 = random_slice(rng, shape[0], max_step, prob_step_negative, prob_start_stop_negative)
            subvector = vector[loc1]
            loc1_cleaned = clean_slice(loc1, shape[0])

            assert np.array_equal(subvector, vector[loc1_cleaned])

            loc2 = random_int(rng, subvector.shape[0])
            element = subvector[loc2]
            joined_loc = index_slice(loc1_cleaned, loc2, shape[0])

            assert np.array_equal(element, vector[joined_loc])

    @pytest.mark.parametrize("loc1,loc2", [
        (
            slice(18, 7, -1),
            slice(8, None, -1),
        ),
        (
            slice(16, None, -1),
            slice(2, 17, 1),
        )
    ], ids=[
        "None stop onto negative step",
        "Numerical stop lands on -1 index",
    ])
    def test_add_slices_named_cases(self,
        vector: np.ndarray,
        shape: tuple[int, ...],
        loc1: slice,
        loc2: slice,
    ):
        subvector = vector[loc1]
        loc1_cleaned = clean_slice(loc1, shape[0])

        assert np.array_equal(subvector, vector[loc1_cleaned])

        subsubvector = subvector[loc2]
        joined_loc = add_indices(loc1_cleaned, loc2, shape[0])

        assert np.array_equal(subsubvector, vector[joined_loc])

    def test_add_slices(self,
        vector: np.ndarray,
        shape: tuple[int, ...],
        rng: random.Generator,
        max_step: int,
        prob_step_negative: float,
        prob_start_stop_negative: float,
    ):
        for _ in range(1000):
            loc1 = random_slice(rng, shape[0], max_step, prob_step_negative, prob_start_stop_negative)
            subvector = vector[loc1]
            loc1_cleaned = clean_slice(loc1, shape[0])

            assert np.array_equal(subvector, vector[loc1_cleaned])

            loc2 = random_slice(rng, subvector.shape[0], max_step, prob_step_negative, prob_start_stop_negative)
            subsubvector = subvector[loc2]
            joined_loc = add_indices(loc1_cleaned, loc2, shape[0])

            assert np.array_equal(subsubvector, vector[joined_loc])


class TestComputeIndices:
    @pytest.mark.parametrize("indices", [
        pytest.param(
            (slice(None, None, 2), 2, slice(1)),
            id="slice"
        ),
        pytest.param(
            (slice(10, None, -1), slice(2), slice(2, None)),
            id="step=-1"
        ),
        pytest.param(
            (slice(3), slice(None, None, 2), slice(2, None)),
            id="step>1"
        ),
        pytest.param(
            (slice(10, None, -2), slice(2), slice(2, None)),
            id="step<-1"
        ),
        pytest.param(
            (0, 1, 2, 3),
            id="element",
            marks=pytest.mark.xfail(reason="Indexing a scalar results in a copy")
        ),
    ])
    def test_single_index_op(self, np_array: np.ndarray, indices: list[Indices]):
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
