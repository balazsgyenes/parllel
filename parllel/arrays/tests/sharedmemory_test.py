import pytest

import multiprocessing as mp
import numpy as np

from parllel.arrays.sharedmemory import SharedMemoryArray

pytest.skip(allow_module_level=True)

@pytest.fixture(autouse=True)
def run_before_tests():
    mp.set_start_method("spawn")
    assert mp.get_start_method() == "spawn", "Setting multiprocessing start method to spawn failed."


def test_sharedmemory_array():
    shape = (5, 5)

    array = SharedMemoryArray(shape=shape, dtype=np.int32)
    array.initialize()

    assert np.count_nonzero(array) == 0

    p = mp.Process(target=fill_array_in_subprocess, args=(array,))
    p.start()
    p.join()

    array = np.asarray(array)
    assert np.all(array == 5)

def fill_array_in_subprocess(array: SharedMemoryArray):
    local_array = array
    local_array[:, :] = 5
