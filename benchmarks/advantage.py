import time
from numpy.random import default_rng

import numpy as np
from numba import njit

from parllel.transforms.advantage import _generalized_advantage_estimation


n_repeat = 100
T, B = 256, 32
max_length = 256
discount = 0.99
lambda_ = 0.95


def generate(rng):
    
    reward = 10 * rng.random((T, B), dtype=np.float32) - 5
    value = 10 * rng.random((T, B), dtype=np.float32) - 5
    done = np.zeros((T, B), dtype=np.bool_)

    for b in range(B):
        end_step = rng.integers(max_length)
        if end_step < T:
            done[end_step, b] = True

    bootstrap_value = 10 * rng.random((B,), dtype=np.float32) - 5

    return reward, value, done, bootstrap_value


def benchmark(func, rng):
    reward, value, done, bootstrap_value = generate(rng)
    advantage = np.zeros_like(reward)
    return_ = np.zeros_like(reward)

    # first run not timed, in case we need to jit
    func(reward, value, done, bootstrap_value,
        discount, lambda_, advantage, return_)

    times = np.zeros(n_repeat, dtype=float)

    for n in range(n_repeat):
        reward, value, done, bootstrap_value = generate(rng)
        start = time.time()
        func(reward, value, done, bootstrap_value,
            discount, lambda_, advantage, return_)
        end = time.time()
        times[n] = (end - start) * 1000000

    print(f"Average time = {times.mean():.3f} +/- {times.std():.3f} ns")
    print()


if __name__ == "__main__":

    seed = 42

    print("Benchmarking Python function")
    rng = default_rng(seed)
    func = _generalized_advantage_estimation
    benchmark(func, rng)

    print("Benchmarking njit")
    rng = default_rng(seed)
    func = njit()(_generalized_advantage_estimation)
    benchmark(func, rng)

    print("Benchmarking njit(fastmath=True)")
    rng = default_rng(seed)
    func = njit(fastmath=True)(_generalized_advantage_estimation)
    benchmark(func, rng)

    print("Benchmarking njit(parallel=True, fastmath=True)")
    rng = default_rng(seed)
    func = njit(parallel=True, fastmath=True)(_generalized_advantage_estimation)
    benchmark(func, rng)
