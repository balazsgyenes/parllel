import time

import numpy as np
from numpy.random import default_rng

from parllel.arrays.array import Array
from parllel.arrays.rotating import RotatingArray
from parllel.arrays.indices import random_location

n_repeat = 100



def benchmark(array, np_array, rng):

    times_lazy_indexing = np.zeros(n_repeat, np.float64)
    times_strict_indexing = np.zeros(n_repeat, np.float64)
    times_np_indexing = np.zeros(n_repeat, np.float64)

    for n in range(n_repeat):

        location = (slice(5), 7, slice(1,-1))

        start = time.perf_counter_ns()

        subarray = array[location]

        after_index = time.perf_counter_ns()

        _ = subarray.shape

        end = time.perf_counter_ns()
        times_lazy_indexing[n] = (after_index - start) / 1000
        times_strict_indexing[n] = (end - start) / 1000

        start = time.perf_counter_ns()

        np_subarray = np_array[location]

        end = time.perf_counter_ns()
        times_np_indexing[n] = (end - start) / 1000

    print(f"Array lazy indexing = {times_lazy_indexing.mean():.3f} +/- "
        f"{times_lazy_indexing.std():.3f} us")
    print(f"Array strict indexing = {times_strict_indexing.mean():.3f} +/- "
        f"{times_strict_indexing.std():.3f} us")
    print(f"Numpy indexing = {times_np_indexing.mean():.3f} +/- "
        f"{times_np_indexing.std():.3f} us")


if __name__ == "__main__":

    seed = 42
    rng = default_rng(seed)

    shape = (10, 9, 8)
    np_array = np.arange(np.prod(shape), dtype=np.float32)
    np_array.shape = shape

    array = Array(shape, np.float32)
    array[...] = np_array

    print("Array:")
    benchmark(array, np_array, rng)

    array = RotatingArray(shape, np.float32)
    array[...] = np_array

    print("RotatingArray:")
    benchmark(array, np_array, rng)
