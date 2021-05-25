import pytest
import numpy as np
from parllel.arrays.array import Array


class TestArray:
    def test_no_args(self):
        with pytest.raises(TypeError):
            _ = Array()

    def test_wrong_dtype(self):
        with pytest.raises(AssertionError):
            shape = (4, 4, 4)
            _ = Array(shape=shape, dtype=list)

    def test_not_initialized(self):
        with pytest.raises(AttributeError):
            shape = (4, 4, 4)
            array = Array(shape=shape, dtype=np.float32)
            array[1, 2, 3] = 5.0

    def test_initialize(self):
        shape = (4, 4, 4)
        array = Array(shape=shape, dtype=np.float32)
        array.initialize()
        assert np.all(array == np.zeros(shape))

    def test_setitem(self):
        shape = (4, 4, 4)
        array = Array(shape=shape, dtype=np.float32)
        array.initialize()

        array[:] = 5.0

        same_values = np.ones(shape, dtype=np.float32)
        same_values[:] = 5.0

        assert np.all(array == same_values)

    def test_getitem(self):
        shape = (4, 4, 4)
        array = Array(shape=shape, dtype=np.float32)
        array.initialize()

        array[1, 2, 3] = 5.0

        assert array[1, 2, 3] == 5.0

    def test_setitem_slice(self):
        shape = (4, 4, 4)
        array = Array(shape=shape, dtype=np.float32)
        array.initialize()

        array[:, slice(1, 3), 3] = 5.0

        same_values = np.zeros(shape, dtype=np.float32)
        same_values[:, slice(1, 3), 3] = 5.0

        assert np.all(array == same_values)

    def test_getitem_slice(self):
        shape = (4, 4, 4)
        array = Array(shape=shape, dtype=np.float32)
        array.initialize()

        assert array[:, slice(1, 3), 3].shape == (4, 2)

    def test_dtype(self):
        array = Array(shape=(4, 4, 4), dtype=np.float32)
        array.initialize()
        array[0, 0, 0] = 5
        assert type(array[0, 0, 0]) == np.float32
