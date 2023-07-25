import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

N_REPEATS = 10000


if __name__ == "__main__":
    shmem = SharedMemory(create=True, size=10000)

    name = shmem.name
    del shmem

    times = np.zeros((N_REPEATS,), dtype=np.float32)

    for i in range(N_REPEATS):
        start = time.perf_counter_ns()
        shmem = SharedMemory(create=False, name=name)
        end = time.perf_counter_ns()

        del shmem

        times[i] = end - start

    shmem = SharedMemory(create=False, name=name)
    shmem.close()
    shmem.unlink()

    print(
        f"Average time over {N_REPEATS} repetitions is {times.mean() / 1000} "
        f"+/- {times.std() / np.sqrt(N_REPEATS) / 1000} us"
    )
