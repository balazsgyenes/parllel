import time

from numba import njit
import numpy as np
from numpy.random import default_rng
from nptyping import NDArray


def generalized_advantage_estimation(
        reward: NDArray[np.float32],
        value: NDArray[np.float32],
        done: NDArray[np.bool_],
        bootstrap_value: NDArray[np.float32],
        discount: float,
        gae_lambda: float,
        out_advantage: NDArray[np.float32],
        out_return: NDArray[np.float32],
    ) -> None:
    """Time-major inputs, optional other dimensions: [T], [T,B], etc.  Similar
    to `discount_return()` but using Generalized Advantage Estimation to
    compute advantages and returns."""
    advantage, return_ = out_advantage, out_return
    not_done = np.logical_not(done)
    advantage[-1] = reward[-1] + discount * bootstrap_value * not_done[-1] - value[-1]
    # for t in reversed(range(len(reward) - 1)): # numba doesn't support reversed
    for t in range(len(reward) - 2, -1, -1): # iterate backwards through time
        delta = reward[t] + discount * value[t + 1] * not_done[t] - value[t]
        advantage[t] = delta + discount * gae_lambda * not_done[t] * advantage[t + 1]
    return_[:] = advantage + value


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
    func = generalized_advantage_estimation
    benchmark(func, rng)

    print("Benchmarking njit")
    rng = default_rng(seed)
    func = njit()(generalized_advantage_estimation)
    benchmark(func, rng)

    print("Benchmarking njit(fastmath=True)")
    rng = default_rng(seed)
    func = njit(fastmath=True)(generalized_advantage_estimation)
    benchmark(func, rng)

    print("Benchmarking njit(parallel=True, fastmath=True)")
    rng = default_rng(seed)
    func = njit(parallel=True, fastmath=True)(generalized_advantage_estimation)
    benchmark(func, rng)
