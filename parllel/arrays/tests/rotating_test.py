import pytest
import numpy as np
from parllel.arrays.rotating import RotatingArray, shift_index


@pytest.fixture(params=[RotatingArray])
def ArrayClass(request):
    return request.param

@pytest.fixture
def shape():
    return (10, 4, 4)

@pytest.fixture(params=[np.float32, np.int32])
def dtype(request):
    return request.param

@pytest.fixture(params=[1, 2], ids=["padding=1", "padding=2"])
def padding(request):
    return request.param

@pytest.fixture
def blank_array(ArrayClass, shape, dtype, padding):
    return ArrayClass(shape=shape, dtype=dtype, padding=padding)

@pytest.fixture
def np_array(shape, dtype):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

@pytest.fixture
def array(blank_array, np_array):
    blank_array[:] = np_array
    return blank_array


class TestRotatingArray:
    def test_negative_padding(self, ArrayClass, shape, dtype, padding):
        with pytest.raises(AssertionError):
            _ = ArrayClass(shape=shape, dtype=dtype, padding=-padding)

    def test_init(self, blank_array, shape, dtype):
        assert blank_array.shape == shape
        assert np.asarray(blank_array).shape == shape
        assert blank_array.dtype == dtype
        assert np.asarray(blank_array).dtype == dtype

    def test_setitem_single(self, array, np_array):
        array[-1] = -7
        array[array.end + 1] = -8

        assert np.array_equal(array, np_array)
        assert np.all(np.asarray(array[-1]) == -7)
        assert np.all(np.asarray(array[array.end + 1]) == -8)

    def test_setitem_slices(self, array, np_array):
        array[:] = -7
        np_array[:] = -7

        assert np.array_equal(array, np_array)
        assert np.all(np.asarray(array[-1]) == 0)
        assert np.all(np.asarray(array[array.end + 1]) == 0)

    def test_setitem_ellipsis(self, array, np_array):
        array[..., 1] = -7
        np_array[..., 1] = -7
        zeros = np.zeros_like(np_array)

        assert np.array_equal(array, np_array)
        assert np.array_equal(array[-1], zeros[0])
        assert np.array_equal(array[array.end + 1], zeros[0])

    def test_setitem_beyond_end(self, blank_array):
        with pytest.raises(IndexError):
            blank_array[blank_array.end + 3] = 1

    def test_setitem_beyond_beginning(self, blank_array):
        with pytest.raises(IndexError):
            blank_array[-3] = 1

    def test_rotate(self, array, np_array, padding):
        array[array.end - 1] = -7
        array[array.end] = -8
        array[array.end + 1] = -9
        if padding >= 2:
            array[array.end + 2] = -10

        array.rotate()
        np_array[-2] = -7
        np_array[-1] = -8
        np_array[0] = -9
        if padding >= 2:
            np_array[1] = -10
        ones = np.ones_like(np_array)

        assert np.array_equal(array, np_array)
        if padding >= 2:
            assert np.array_equal(array[-2], ones[0] * -7)
        assert np.array_equal(array[-1], ones[0] * -8)
        assert np.array_equal(array[0], ones[0] * -9)
        if padding >= 2:
            assert np.array_equal(array[1], ones[0] * -10)

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
