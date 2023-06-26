import time

import numpy as np
from numpy.random import default_rng

from parllel.arrays.indices import add_locations, compute_indices, init_location
from parllel.arrays.tests.indices_test import random_location


N_REPEAT = 1000
BASE_SHAPE = (20, 20, 20, 20)


def benchmark_multiple_indices(rng):

    times_add_indices = np.zeros(N_REPEAT, dtype=float)
    times_index = np.zeros(N_REPEAT, dtype=float)
    times_compute_indices = np.zeros(N_REPEAT, dtype=float)

    for n in range(N_REPEAT):

        np_array = rng.random(BASE_SHAPE, dtype=np.float32)

        current_location = init_location(BASE_SHAPE)

        loc1 = random_location(rng, np_array.shape, max_step=3, prob_start_stop_negative=0.5, prob_step_negative=0.5)
        subarray = np_array[loc1]
        current_location = add_locations(current_location, loc1, np_array.shape)

        if not subarray.shape:
            continue  # if we have indexed a single element already, skip

        loc2 = random_location(rng, subarray.shape, max_step=3, prob_start_stop_negative=0.5, prob_step_negative=0.5)


        start = time.perf_counter_ns()

        current_location = add_locations(current_location, loc2, np_array.shape)

        end = time.perf_counter_ns()
        times_add_indices[n] += (end - start) / 1000


        start = time.perf_counter_ns()

        subarray = subarray[loc2]

        end = time.perf_counter_ns()
        times_index[n] += (end - start) / 1000


        start = time.perf_counter_ns()

        computed_located = compute_indices(np_array, subarray)

        end = time.perf_counter_ns()
        times_compute_indices[n] += (end - start) / 1000


    print(f"Indexing = {times_index.mean():.3f} +/- {times_index.std():.3f} us")
    print(f"Adding indices = {times_add_indices.mean():.3f} +/- {times_add_indices.std():.3f} us")
    print(f"Computing indices = {times_compute_indices.mean():.3f} +/- {times_compute_indices.std():.3f} us")


if __name__ == "__main__":

    seed = 42

    rng = default_rng(seed)
    benchmark_multiple_indices(rng)
