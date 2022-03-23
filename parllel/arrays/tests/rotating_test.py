import functools

import pytest
import numpy as np

from parllel.arrays.rotating import RotatingArray, shift_index


@pytest.fixture(params=[RotatingArray], scope="module")
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

@pytest.fixture(scope="module")
def gen_blank_array(ArrayClass, shape, dtype, padding):
    def _gen_blank_array():
        return ArrayClass(shape=shape, dtype=dtype, padding=padding)
    return _gen_blank_array

@pytest.fixture
def blank_array(gen_blank_array):
    return gen_blank_array()

@pytest.fixture(scope="module")
def gen_np_array(shape, dtype):
    def _gen_np_array():
        return np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    return _gen_np_array

@pytest.fixture
def np_array(gen_np_array):
    return gen_np_array()

@pytest.fixture(scope="module")
def gen_previous_region(gen_np_array, padding):
    def _gen_previous_region():
        np_array = gen_np_array()
        # flip data in trailing dims
        return np_array[:padding, ::-1, ::-1]
    return _gen_previous_region

@pytest.fixture
def previous_region(gen_previous_region):
    return gen_previous_region()

@pytest.fixture(scope="module")
def gen_next_region(gen_np_array, padding):
    def _gen_next_region():
        np_array = gen_np_array()
        # flip data in trailing dims, scale to make unique
        return np_array[:padding, ::-1, ::-1] * 2
    return _gen_next_region

@pytest.fixture
def next_region(gen_next_region):
    return gen_next_region()

@pytest.fixture(scope="module")
def gen_array(gen_blank_array, gen_np_array, padding, gen_previous_region, gen_next_region):
    def _gen_array():
        array = gen_blank_array()
        np_array = gen_np_array()
        previous_region = gen_previous_region()
        next_region = gen_next_region()
        array[:] = np_array
        array[-padding:0] = previous_region
        end = array.end
        array[(end + 1):(end + padding + 1)] = next_region
        return array
    return _gen_array

@pytest.fixture()
def array(gen_array):
    return gen_array()


class TestRotatingArray:
    def test_negative_padding(self, ArrayClass, shape, dtype, padding):
        with pytest.raises(AssertionError):
            _ = ArrayClass(shape=shape, dtype=dtype, padding=-padding)

    def test_init(self, blank_array, shape, dtype):
        assert blank_array.shape == shape
        assert np.asarray(blank_array).shape == shape
        assert blank_array.dtype == dtype
        assert np.asarray(blank_array).dtype == dtype

    def test_setitem_single(self, array, np_array, next_region):
        array[-1] = -7
        array[array.end + 1] = -8
        ones = np.ones_like(next_region[0])

        assert np.array_equal(array, np_array)
        assert np.array_equal(array[-1], ones * -7)
        np.array_equal(array[array.end + 1], ones * -8)

    def test_setitem_slices(self, array, shape, dtype, np_array):
        array[-1:1] = -7
        ones = np.ones((2,) + shape[1:], dtype)
        assert np.array_equal(array[-1:1], ones * -7)
        assert np.array_equal(array[1:], np_array[1:])

    def test_setitem_all(self, array, padding, previous_region, next_region):
        array[:] = -7

        assert np.array_equal(array[-padding:0], previous_region)
        end = array.end
        assert np.array_equal(array[(end + 1):(end + padding + 1)],
                              next_region)

    def test_setitem_ellipsis(self, array, padding, previous_region, next_region):
        array[..., 1] = -7

        assert np.array_equal(array[-padding:0], previous_region)
        end = array.end
        assert np.array_equal(array[(end + 1):(end + padding + 1)],
                              next_region)

    def test_setitem_beyond_end(self, blank_array, padding):
        with pytest.raises(IndexError):
            blank_array[blank_array.end + padding + 1] = 1

    def test_setitem_beyond_beginning(self, blank_array, padding):
        with pytest.raises(IndexError):
            blank_array[-padding - 1] = 1

    def test_getitem(self, array, np_array, previous_region):
        array = array[-1:]
        assert array.shape == (11, 4, 4)
        assert np.asarray(array).shape == (11, 4, 4)
        values = np.concatenate((previous_region[-1:], np_array))
        assert np.array_equal(array, values)

        array = array[2]
        assert array.shape == (4, 4)
        assert np.asarray(array).shape == (4, 4)
        assert np.array_equal(array, np_array[1])

    def test_getitem_consecutively(self, array, np_array):
        array = array[-1:]
        array = array[2]
        assert array.shape == (4, 4)
        assert np.asarray(array).shape == (4, 4)
        assert np.array_equal(array, np_array[1])

    def test_rotate(self, array, np_array, padding, next_region):
        array.rotate()
        assert np.array_equal(array[:padding], next_region)
        assert np.array_equal(array[padding:], np_array[padding:])

    def test_indexhistory(self, array):
        assert array.index_history == ()

        array = array[-1:]
        assert array.index_history == (slice(-1, None), )

        array = array[1]
        assert array.index_history == (slice(-1, None), 1)

        array = array[1:3]
        assert array.index_history == (slice(-1, None), 1, slice(1, 3))

        array = array[-1, ::2]
        assert array.index_history == (
            slice(-1, None),
            1,
            slice(1, 3),
            (-1, slice(None, None, 2)),
        )

    def test_array_reconstruction(self, gen_array):
        array = gen_array()
        # same as Array test, just start with negative index here
        array = array[-1:]
        array = array[:]
        array = array[0:2]
        array = array[:, -2:]

        array2 = gen_array()
        # apply array's index history to array2
        array2 = functools.reduce(
            lambda buf, index: buf[index],
            array.index_history,
            array2
        )

        assert np.array_equal(array, array2)
        assert all(el1 == el2 for el1, el2
            in zip(array.index_history, array2.index_history))

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
