import pytest
import numpy as np

# reuse fixtures from array_test
from array_test import ArrayClass, shape, dtype, storage, padding, blank_array, np_array, array


@pytest.fixture(params=[20], ids=["2X_size"], scope="module")
def full_size(request):
    return request.param


class TestFullSize:
    def test_nonmultiple_fullsize(self, ArrayClass, shape, dtype, storage):
        with pytest.raises(ValueError):
            _ = ArrayClass(batch_shape=shape, dtype=dtype, storage=storage, full_size=15)

    def test_rotate(self, array, np_array, padding):
        array.rotate()
        array[:] = np_array * 2
        # in case there is padding, ensure that the value expected in the
        # beginning is copied into the padding
        array[(array.last + 1):(array.last + padding + 1)] = np_array[:padding]
        array.rotate()

        assert np.array_equal(array, np_array)
        array.rotate()
        assert np.array_equal(array, np_array * 2)

    def test_full(self, array, np_array):
        array.rotate()
        array[:] = np_array * 2

        full_np_array = np.concatenate((np_array, np_array * 2))
        assert np.array_equal(array.full, full_np_array)

    def test_write_to_full(self, array, np_array):
        array.full[10:20] = np_array * 2

        array.rotate()
        assert np.array_equal(array, np_array * 2)

    def test_write_past_bounds(self, array, np_array):
        array[10:20] = np_array * 2

        array.rotate()
        assert np.array_equal(array, np_array * 2)

    def test_write_before_bounds(self, array, np_array):
        array.rotate()
        array[-10:0] = np_array * 2

        assert np.array_equal(array.full[0:10], np_array * 2)
