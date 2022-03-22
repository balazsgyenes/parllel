import pytest
import numpy as np
from parllel.arrays.rotating import RotatingArray, shift_index


@pytest.fixture
def shape():
    return (10, 4, 3)

@pytest.fixture
def dtype():
    return np.float32

@pytest.fixture
def padding():
    return 2

@pytest.fixture
def blank_array(shape, dtype, padding):
    return RotatingArray(shape=shape, dtype=dtype, padding=padding)

@pytest.fixture
def np_array(shape, dtype):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

@pytest.fixture
def array(blank_array, np_array):
    blank_array[:] = np_array
    return blank_array


class TestRotatingArray:
    def test_init(self, blank_array, shape, dtype):
        assert blank_array.shape == shape
        assert blank_array.dtype == dtype

    def test_negative_padding(self):
        with pytest.raises(AssertionError):
            _ = RotatingArray(shape=(10, 4, 4, 3), dtype=np.uint8, padding=-2)

    def test_padded_indices(self, array, np_array):
        array[-1] = 0.5
        array[11] = 0.5

        assert np.array_equal(array, np_array)
        assert np.all(np.asarray(array[-1]) == 0.5)
        assert np.all(np.asarray(array[11]) == 0.5)
        assert np.all(np.asarray(array[-2]) == 0)

    def test_out_of_bounds(self, blank_array):
        with pytest.raises(IndexError):
            blank_array[12] = 1

    def test_colon_slice(self, array, np_array):
        array[:] = 1.5
        np_array[:] = 1.5

        assert np.array_equal(array, np_array)
        assert np.all(np.asarray(array[-1]) == 0)
        assert np.all(np.asarray(array[10]) == 0)

    def test_ellipsis_slice(self, array, np_array):
        array[..., 1] = 2.5
        np_array[..., 1] = 2.5
        zeros = np.zeros_like(np_array)

        assert np.array_equal(array, np_array)
        assert np.array_equal(array[-1], zeros[0])
        assert np.array_equal(array[10], zeros[0])

    def test_rotate(self, array, np_array):
        array[array.end - 1] = 0.5
        array[array.end] = 1.5
        array[array.end + 1] = 2.5
        array[array.end + 2] = 3.5

        array.rotate()
        np_array[-2] = 0.5
        np_array[-1] = 1.5
        np_array[0] = 2.5
        np_array[1] = 3.5
        ones = np.ones_like(np_array)

        assert np.array_equal(array, np_array)
        assert np.array_equal(array[-2], ones[0] * 0.5)
        assert np.array_equal(array[-1], ones[0] * 1.5)
        assert np.array_equal(array[0], ones[0] * 2.5)
        assert np.array_equal(array[1], ones[0] * 3.5)

    def test_previous_array(self):
        pass

    def test_shift_integer(self):
        assert shift_index(4, 2) == (6,)

    def test_shift_slice(self):
        assert shift_index(slice(1, 5, 2), 2) == (slice(3, 7, 2),)
    
    def test_shift_ellipsis(self):
        assert shift_index(..., 2) == (slice(2, -2), Ellipsis)
