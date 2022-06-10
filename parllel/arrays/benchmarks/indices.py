import time

import numpy as np
from numpy.random import default_rng

from parllel.arrays.indices import add_indices, compute_indices


n_repeat = 100
base_shape = (10, 10, 10, 10)

single = (4, 7, 3)

multiple = [
    (2, slice(None)),
    (slice(1, None), slice(2, -1), 5),
    (6, slice(None, 4)),
]


def random_indices(rng):
    pass

def benchmark_multiple_indices(rng):

    times_add_indices = np.zeros(n_repeat, dtype=float)
    times_index = np.zeros(n_repeat, dtype=float)
    times_compute_indices = np.zeros(n_repeat, dtype=float)

    for n in range(n_repeat):

        np_array = rng.random(base_shape, dtype=np.float32)
        current_indices = []
        for location in multiple:
            start = time.perf_counter_ns()

            current_indices = add_indices(base_shape, current_indices, location)

            end = time.perf_counter_ns()
            times_add_indices[n] += (end - start) / 1000


            start = time.perf_counter_ns()

            next_array = np_array[location]

            end = time.perf_counter_ns()
            times_index[n] += (end - start) / 1000


            start = time.perf_counter_ns()

            computed_indices = compute_indices(np_array, np_array)

            end = time.perf_counter_ns()
            times_compute_indices[n] += (end - start) / 1000

            # assert computed_indices == current_indices
            np_array = next_array

    print(f"Adding indices = {times_add_indices.mean():.3f} +/- {times_add_indices.std():.3f} us")
    print(f"Indexing = {times_index.mean():.3f} +/- {times_index.std():.3f} us")
    print(f"Computing indices = {times_compute_indices.mean():.3f} +/- {times_compute_indices.std():.3f} us")


if __name__ == "__main__":

    seed = 42

    rng = default_rng(seed)
    benchmark_multiple_indices(rng)
