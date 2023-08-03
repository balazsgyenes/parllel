# fmt: off
import math
from itertools import islice
from typing import Sequence

import numpy as np
import numpy.random as random
import pytest

from parllel.arrays.indices import (Index, Location, batch_dims_from_location,
                                    clean_slice, compose_indices,
                                    compose_locations, compose_slice_with_int,
                                    compose_slices, compute_indices,
                                    init_location, predict_copy_on_index,
                                    shape_from_location)

# fmt: on
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

    start = (
        start - size
        if (rng.random() < prob_start_stop_negative and start is not None)
        else start
    )
    stop = (
        stop - size
        if (
            rng.random() < prob_start_stop_negative
            and stop is not None
            and stop < size  # if stop==size, cannot be converted to negative index
        )
        else stop
    )

    return slice(start, stop, step)


def random_location(
    rng: random.Generator,
    shape: tuple[int, ...],
    max_step: int = MAX_STEP,
    prob_start_stop_negative: float = PROB_INDEX_NEGATIVE,
    prob_step_negative: float = PROB_STEP_NEGATIVE,
) -> Location:
    return tuple(
        (
            random_slice(
                rng,
                size,
                max_step,
                prob_step_negative,
                prob_start_stop_negative,
            )
            if rng.random() < PROB_SLICE
            else random_int(rng, size)
        )
        for size in islice(
            shape,
            rng.integers(1, high=len(shape), endpoint=True),
        )
    )


# fmt: off
@pytest.fixture(params=[
    (20, 22, 24, 26)
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
    def test_add_locations(self,
        np_array: np.ndarray,
        rng: random.Generator,
        max_step: int,
        prob_step_negative: float,
        prob_start_stop_negative: float,
    ):
        for _ in range(1000):
            loc1 = random_location(rng, np_array.shape, max_step, prob_step_negative, prob_start_stop_negative)
            subarray = np_array[loc1]
            loc1_cleaned = compose_locations(init_location(np_array.shape), loc1, np_array.shape)  # clean location

            assert np.array_equal(subarray, np_array[tuple(loc1_cleaned)])

            if not subarray.shape:
                continue  # if we have indexed a single element already, skip
            
            loc2 = random_location(rng, subarray.shape, max_step, prob_step_negative, prob_start_stop_negative)
            subsubarray = subarray[loc2]
            joined_loc = compose_locations(loc1_cleaned, loc2, np_array.shape)
            
            assert np.array_equal(subsubarray, np_array[tuple(joined_loc)])

    @pytest.mark.parametrize("loc1,loc2", [
        (
            slice(None),
            np.array([2, 10, 6, 17, 6]),
        ),
        (
            slice(None),
            (np.array([0, 5, 3, 2, 9]), np.array([14, 2, 7, 4, 8])),
        ),
        (
            slice(2, -2),
            np.array([4, 13, 12, 11, 8]),
        ),
        (
            slice(3, -3),
            (np.array([1, 3, 12, 10, 7]), np.array([11, 2, 7, 4, 8])),
        ),
        (
            (10, slice(None, None, 2)),
            (np.array([2, 1, -2, -4, 8]), np.array([2, -8, 3, 5, 5])),
        ),
        (
            slice(None, None, -2),
            (np.array([2, 3, -2, -4, 7]), np.array([2, -5, 3, 4, 1])),
        ),
        (
            (slice(1, -1), slice(2, -2), slice(3, -3), slice(4, -4)),
            (np.array([9, -9, -7, -11]), np.array([-13, -14, -8, 0]), np.array([6, -11, 8, -9]), np.array([-6, 5, 5, -8])),
        ),
        (
            slice(10, None, -2),
            (
                np.array([[-2, -5, -5, -4],
                          [-2, -2,  3, -3],
                          [ 3,  2, -2, -1],
                          [ 3,  4,  2,  4]]),
                np.array([[-1,   8,   0,  18],
                          [ 3,   5,  10,   3],
                          [-5,  -7,  18,  -6],
                          [ 1, -11,   9,   3]]),
            ),
        ),
    ], ids=[
        "single 1D array",
        "two 1D arrays at once",
        "single 1D array onto a slice",
        "two 1D arrays onto a slice",
        "two 1D arrays onto a slice with a step",
        "two 1D arrays onto a slice with a negative step",
        "4 slices followed by 4 1D arrays",
        "two 2D arrays onto a slice with a negative step",
    ])
    def test_add_locations_with_arrays(self,
        np_array: np.ndarray,
        loc1: Location,
        loc2: Location,
    ):
        subarray = np_array[loc1]
        loc1_cleaned = compose_locations(init_location(np_array.shape), loc1, np_array.shape)  # clean location

        assert np.array_equal(subarray, np_array[tuple(loc1_cleaned)])

        subsubarray = subarray[loc2]
        joined_loc = compose_locations(loc1_cleaned, loc2, np_array.shape)

        assert np.array_equal(subsubarray, np_array[tuple(joined_loc)])

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
        index_history: Sequence[Location],
    ):
        subarray = np_array
        joined_loc = init_location(np_array.shape)

        for indices in index_history:
            subarray = subarray[indices]
            joined_loc = compose_locations(joined_loc, indices, np_array.shape)

        assert np.array_equal(subarray, np_array[tuple(joined_loc)])

    def test_add_slices(self,
        vector: np.ndarray,
        rng: random.Generator,
        max_step: int,
        prob_step_negative: float,
        prob_start_stop_negative: float,
    ):
        for _ in range(1000):
            slice1 = random_slice(rng, vector.shape[0], max_step, prob_step_negative, prob_start_stop_negative)
            subvector = vector[slice1]
            slice1_cleaned = clean_slice(slice1, vector.shape[0])

            assert np.array_equal(subvector, vector[slice1_cleaned])

            slice2 = random_slice(rng, subvector.shape[0], max_step, prob_step_negative, prob_start_stop_negative)
            subsubvector = subvector[slice2]
            joined_slice = compose_slices(slice1_cleaned, slice2, vector.shape[0])

            assert np.array_equal(subsubvector, vector[joined_slice])

    @pytest.mark.parametrize("slice1,slice2", [
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
        slice1: slice,
        slice2: slice,
    ):
        subvector = vector[slice1]
        slice1_cleaned = clean_slice(slice1, vector.shape[0])

        assert np.array_equal(subvector, vector[slice1_cleaned])

        subsubvector = subvector[slice2]
        joined_slice = compose_slices(slice1_cleaned, slice2, vector.shape[0])

        assert np.array_equal(subsubvector, vector[joined_slice])

    def test_cannot_add_after_advanced_index(self,
        np_array: np.ndarray,
    ):
        loc1 = np.array([3, -9, -6, 12, -8, 1, 7, 3, 10, 19, 1, 0, -3, -7, 8, -4, 7, 19, 3, -1])
        subarray = np_array[loc1]
        loc1_cleaned = compose_locations(init_location(np_array.shape), loc1, np_array.shape)  # clean location

        assert np.array_equal(subarray, np_array[tuple(loc1_cleaned)])

        loc2 = (slice(None), np.array([18, -3, 16, -6, -3, 3, 18, -5, -1, 1, 1, 5, 18, 0, -4, 12, 13, 1, 1, -6]))

        with pytest.raises(IndexError):
            _ = compose_locations(loc1_cleaned, loc2, np_array.shape)

    def test_index_scalar(self,
        np_array,
        rng: random.Generator,
    ):
        joined_location = init_location(np_array.shape)

        loc = tuple(random_int(rng, size) for size in np_array.shape)
        element = np_array[loc]
        joined_location = compose_locations(joined_location, loc, np_array.shape)

        assert np.array_equal(element, np_array[tuple(joined_location)])
        assert element.shape == ()

        loc = (Ellipsis,)
        joined_location = compose_locations(joined_location, loc, np_array.shape)

        assert np.array_equal(element, np_array[tuple(joined_location)])
        assert element.shape == ()

        loc = ()  # empty tuple
        joined_location = compose_locations(joined_location, loc, np_array.shape)

        assert np.array_equal(element, np_array[tuple(joined_location)])
        assert element.shape == ()

        loc = slice(None)
        with pytest.raises(IndexError):
            _ = compose_locations(joined_location, loc, np_array.shape)

    def test_index_slice(self,
        vector: np.ndarray,
        rng: random.Generator,
        max_step: int,
        prob_step_negative: float,
        prob_start_stop_negative: float,
    ):
        for _ in range(1000):
            slice1 = random_slice(rng, vector.shape[0], max_step, prob_step_negative, prob_start_stop_negative)
            subvector = vector[slice1]
            slice1_cleaned = clean_slice(slice1, vector.shape[0])

            assert np.array_equal(subvector, vector[slice1_cleaned])

            index2 = random_int(rng, subvector.shape[0])
            element = subvector[index2]
            global_index = compose_slice_with_int(slice1_cleaned, index2, vector.shape[0])

            assert np.array_equal(element, vector[global_index])

    @pytest.mark.parametrize("index_history", [
        pytest.param([slice(21, 30)], id="Slice too positive"),
        pytest.param([slice(15, 25)], id="Slice end outside of range"),
        pytest.param([slice(-5, 25)], id="Slice end outside of range, negative start"),
        pytest.param([slice(-25, -15)], id="Slice start outside of range"),
        pytest.param([slice(-30, -21)], id="Slice too negative"),
        pytest.param([slice(-21, -30, -1)], id="Slice too negative, negative step"),
        pytest.param([slice(None, None, -1), slice(-21, -30, -1)], id="Slice too negative, negative step, onto negative step"),
        pytest.param([slice(10, 10)], id="Empty slice"),
    ])
    def test_out_of_bounds_slice(self,
        vector: np.ndarray,
        index_history: Sequence[Index],
    ):

        subvector = vector
        joined_loc = clean_slice(slice(None), vector.shape[0])

        for indices in index_history:
            subvector = subvector[indices]
            joined_loc = compose_indices(joined_loc, indices, vector.shape[0])

        assert np.array_equal(subvector, vector[joined_loc])

    @pytest.mark.parametrize("loc", [
        pytest.param((slice(3), 25), id="Integer too positive"),
        pytest.param((slice(None, None, -1), -25), id="Integer too negative"),
    ])
    def test_out_of_bounds_integer(self, shape, loc):
        with pytest.raises(IndexError):
            _ = compose_locations(init_location(shape), loc, shape)

    @pytest.mark.parametrize("loc", [
        pytest.param(
            (slice(None), 5, -5, slice(5, -5), slice(4, None)),
            id="Trailing extra slice",
        ),
        pytest.param(
            (slice(None), 5, -5, slice(5, -5), 0),
            id="Trailing extra integer",
        ),
    ])
    def test_too_many_indices(self, shape, loc):
        with pytest.raises(IndexError):
            _ = compose_locations(init_location(shape), loc, shape)


class TestComputeIndices:
    @pytest.mark.parametrize("location", [
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
    def test_single_index_op(self, np_array: np.ndarray, location: Location):
        np_subarray = np_array[location]
        location = compute_indices(np_array, np_subarray)
        assert np.array_equal(np_subarray, np_array[tuple(location)])


class TestPredictCopyOnIndex:
    def test_predict_copy_on_index(self):
        assert predict_copy_on_index((2,3,4),(0,0,0))
        assert not predict_copy_on_index((2,3,4), (0,1))
        assert not predict_copy_on_index((9,3,2), (slice(4,5),0,0))


class TestShapeFromLocation:
    def test_shape_from_location(self,
        np_array: np.ndarray,
        rng: random.Generator,
        max_step: int,
        prob_step_negative: float,
        prob_start_stop_negative: float,
    ):
        for _ in range(1000):
            loc1 = random_location(rng, np_array.shape, max_step, prob_step_negative, prob_start_stop_negative)
            subarray = np_array[loc1]
            loc1_cleaned = compose_locations(init_location(np_array.shape), loc1, np_array.shape)  # clean location

            if not (subarray.shape == shape_from_location(loc1_cleaned, np_array.shape)):
                _ = compose_locations(init_location(np_array.shape), loc1, np_array.shape)

            assert subarray.shape == shape_from_location(loc1_cleaned, np_array.shape)

            if not subarray.shape:
                continue  # if we have indexed a single element already, skip
            
            loc2 = random_location(rng, subarray.shape, max_step, prob_step_negative, prob_start_stop_negative)
            subsubarray = subarray[loc2]
            joined_loc = compose_locations(loc1_cleaned, loc2, np_array.shape)
            
            if not (subsubarray.shape == shape_from_location(joined_loc, np_array.shape)):
                _ = compose_locations(loc1_cleaned, loc2, np_array.shape)

            assert subsubarray.shape == shape_from_location(joined_loc, np_array.shape)

    @pytest.mark.parametrize("loc1,loc2", [
        (
            (7, slice(None)),
            (np.array([2, 10, 6, 17, 6]), 5),
        ),
        (
            slice(2, None),
            (np.array([0, 5, 3, 2, 9]), -10, np.array([14, 2, 7, 4, 8])),
        ),
        (
            slice(2, None),
            (np.array([0, 5, 3, 2, 9]), slice(5, -5), np.array([14, 2, 7, 4, 8])),
        ),
        (
            (10, slice(2, -2)),
            np.array([4, 13, 12, 11, 8]),
        ),
        (
            slice(3, -3),
            (10, np.array([1, 3, 12, 10, 7]), np.array([11, 2, 7, 4, 8])),
        ),
        (
            (10, slice(None, None, 2)),
            (np.array([2, 1, -2, -4, 8]), np.array([2, -8, 3, 5, 5])),
        ),
        (
            slice(None, None, -2),
            (np.array([2, 3, -2, -4, 7]), np.array([2, -5, 3, 4, 1])),
        ),
        (
            (slice(1, -1), slice(2, -2), slice(3, -3), slice(4, -4)),
            (np.array([9, -9, -7, -11]), np.array([-13, -14, -8, 0]), np.array([6, -11, 8, -9]), np.array([-6, 5, 5, -8])),
        ),
        (
            slice(10, None, -2),
            (
                np.array([[-2, -5, -5, -4],
                          [-2, -2,  3, -3],
                          [ 3,  2, -2, -1],
                          [ 3,  4,  2,  4]]),
                np.array([[-1,   8,   0,  18],
                          [ 3,   5,  10,   3],
                          [-5,  -7,  18,  -6],
                          [ 1, -11,   9,   3]]),
            ),
        ),
    ], ids=[
        "single 1D array onto an integer and a slice",
        "two 1D arrays with an integer between them",
        "two 1D arrays with a slice between them",
        "single 1D array onto a slice",
        "two 1D arrays onto a slice",
        "two 1D arrays onto a slice with a step",
        "two 1D arrays onto a slice with a negative step",
        "4 slices followed by 4 1D arrays",
        "two 2D arrays onto a slice with a negative step",
    ])
    def test_shape_from_location_index_array(self,
        np_array: np.ndarray,
        loc1: Location,
        loc2: Location,
    ):
        subarray = np_array[loc1]
        loc1_cleaned = compose_locations(init_location(np_array.shape), loc1, np_array.shape)  # clean location

        assert subarray.shape == shape_from_location(loc1_cleaned, np_array.shape)

        subsubarray = subarray[loc2]
        joined_loc = compose_locations(loc1_cleaned, loc2, np_array.shape)

        assert subsubarray.shape == shape_from_location(joined_loc, np_array.shape)

    @pytest.mark.parametrize("location", [
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
    def test_shape_from_location_named_cases(self, np_array, location):
        subarray = np_array[location]
        loc_cleaned = compose_locations(init_location(np_array.shape), location, np_array.shape)  # clean location
        assert subarray.shape == shape_from_location(loc_cleaned, np_array.shape)

    @pytest.mark.parametrize("n_batch_dims,current_batch_dims", [
        (5, 3),
        (4, 2),
        (3, 2),
        (2, 1),
        (1, 0),
        (0, 0),
    ])
    def test_batch_dims_from_location_standard(self, n_batch_dims, current_batch_dims):
        location = [4, slice(3, 4), slice(5, None, -1), 6, slice(None)]
        assert batch_dims_from_location(location, n_batch_dims) == current_batch_dims

    @pytest.mark.parametrize("n_batch_dims,current_batch_dims", [
        (6, 3),
        (5, 2),
        (4, 2),
        (3, 2),
        (2, 1),
        (1, 1),
        (0, 0),
    ])
    def test_batch_dims_from_location_advanced(self, rng, n_batch_dims, current_batch_dims):
        location = [
            rng.integers(20, size=(10,)),
            rng.integers(20, size=(10,)),
            slice(9, None),
            rng.integers(20, size=(10,)),
            9,
            slice(8, 1, -2),
        ]
        assert batch_dims_from_location(location, n_batch_dims) == current_batch_dims
