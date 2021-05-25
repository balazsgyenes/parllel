import pytest
import numpy as np
from parllel.arrays.rotating import RotatingArray, shift_index

class TestRotatingArray:
    def test_init(self):
        array = RotatingArray(shape=(10, 4, 4, 3), dtype=np.uint8, padding=2)

        assert array._apparent_shape == (10, 4, 4, 3)
        assert array._shape == (14, 4, 4, 3)


    def test_wrong_padding(self):
        with pytest.raises(AssertionError):
            _ = RotatingArray(shape=(10, 4, 4, 3), dtype=np.uint8, padding=-2)


    def test_uninitialized_rotate(self):
        with pytest.raises(AttributeError):
            array = RotatingArray(shape=(10, 4, 4, 3), dtype=np.uint8, padding=2)
            array.rotate()


    def test_padded_indices(self):
        array = RotatingArray(shape=(10, 4, 4, 3), dtype=np.uint8, padding=2)
        array.initialize()
        array[-1, ...] = 1
        array[11, ...] = 1

        assert np.all(array[-1, ...] != array[10, ...])
        assert np.all(array[11, ...] != array[-2, ...])


    def test_out_of_bounds(self):
        with pytest.raises(IndexError):
            array = RotatingArray(shape=(10, 4, 4, 3), dtype=np.uint8, padding=2)
            array.initialize()
            array[12, ...] = 1


    def test_rotate(self):
        array = RotatingArray(shape=(10, 4, 4, 3), dtype=np.int32, padding=2)
        array.initialize()

        array[-1, ...] = -1
        array[-2, ...] = -2

        array[9, ...] = 1
        array[10, ...] = 2
        array[11, ...] = 3

        array.rotate()

        assert np.all(array[-2] == np.ones((4, 4, 3)))
        assert np.all(array[-1] == np.ones((4, 4, 3)) * 2)
        assert np.all(array[0] == np.ones((4, 4, 3)) * 3)


def test_shift_index():
    assert shift_index(4, 2) == 6
    assert shift_index(slice(1, 5, 2), 2) == slice(3, 7, 2)
    assert shift_index(..., 2) == slice(2, -2)