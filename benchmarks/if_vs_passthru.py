import time

import numpy as np


if __name__ == "__main__":

    n_loops = int(1e6)

    batch_buffer = np.zeros((1000,), dtype=np.float64)
    obs_transform = lambda x, t: x

    start = time.perf_counter_ns()

    for t in range(n_loops):
        batch_buffer = obs_transform(batch_buffer, t)

    end = time.perf_counter_ns()

    print(f"With pass-thru lambda: {(end - start) / n_loops} ns")


    obs_transform = None

    start = time.perf_counter_ns()

    for _ in range(n_loops):
        if obs_transform is not None:
            batch_buffer = obs_transform(batch_buffer)

    end = time.perf_counter_ns()

    print(f"With if statement: {(end - start) / n_loops} ns")
