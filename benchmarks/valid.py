import time

from numba import njit
import numpy as np
from numpy.random import default_rng
from nptyping import NDArray


def valid_from_done_1(done: NDArray[np.bool_], out_valid: NDArray[np.bool_]):
    valid = out_valid
    valid[0] = True # first time step is always valid
    for t in range(1, len(valid)):
        valid[t] = np.logical_and(valid[t-1], np.logical_not(done[t-1]))
    return valid


def valid_from_done_2(done: NDArray[np.bool_], out_valid: NDArray[np.bool_]):
    valid = out_valid
    valid[0] = True # first time step is always valid
    not_done = np.logical_not(done)
    for t in range(1, len(valid)):
        # valid if previous step was valid and not done yet
        valid[t] = np.logical_and(valid[t-1], not_done[t-1])
    return valid


n_repeat = 100
T, B = 256, 32
max_length = 1000


def generate(rng):
    
    done = np.zeros((T, B), dtype=np.bool_)

    for b in range(B):
        end_step = rng.integers(max_length)
        if end_step < T:
            done[end_step, b] = True

    return done


def benchmark(func, rng):
    done = generate(rng)
    valid = np.zeros_like(done)

    # first run not timed, in case we need to jit
    func(done, valid)

    times = np.zeros(n_repeat, dtype=float)

    for n in range(n_repeat):
        done = generate(rng)
        start = time.time()
        func(done, valid)
        end = time.time()
        times[n] = (end - start) * 1000000

    print(f"Average time = {times.mean():.3f} +/- {times.std():.3f} ns")
    print()


if __name__ == "__main__":

    seed = 42 * 42

    print("Benchmarking Python function (version 1")
    rng = default_rng(seed)
    func = valid_from_done_1
    benchmark(func, rng)

    print("Benchmarking Python function (version 1")
    rng = default_rng(seed)
    func = valid_from_done_2
    benchmark(func, rng)

    print("Benchmarking njit (version 1)")
    rng = default_rng(seed)
    func = njit()(valid_from_done_1)
    benchmark(func, rng)

    print("Benchmarking njit (version 2)")
    rng = default_rng(seed)
    func = njit()(valid_from_done_2)
    benchmark(func, rng)

    print("Benchmarking njit(fastmath=True) (version 1)")
    rng = default_rng(seed)
    func = njit(fastmath=True)(valid_from_done_1)
    benchmark(func, rng)

    print("Benchmarking njit(fastmath=True) (version 2)")
    rng = default_rng(seed)
    func = njit(fastmath=True)(valid_from_done_2)
    benchmark(func, rng)
