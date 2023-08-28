import time

from numba import njit
import numpy as np
from numpy import random

def generate(rng, shape):
    valid = np.zeros(shape, dtype=np.bool_)
    done = rng.random(shape) < 0.05
    return valid, done

def naive(valid, done):
    for t in range(valid.shape[0] - 1):
        valid[t + 1] = np.logical_and(valid[t], np.logical_not(done[t]))

def using_output_arg(valid, done):
    for t in range(valid.shape[0] - 1):
        np.logical_and(valid[t], np.logical_not(done[t]), out=valid[t + 1])

def no_intermediates(valid, done):
    for t in range(valid.shape[0] - 1):
        np.logical_not(done[t], out=valid[t + 1])
        np.logical_and(valid[t], valid[t + 1], out=valid[t + 1])

def benchmark(func, rng, shape, n_repeat):
    valid, done = generate(rng, shape)

    # first run not timed, in case we need to jit
    func(valid, done)

    times = np.zeros(n_repeat, dtype=float)

    for n in range(n_repeat):
        valid, done = generate(rng, shape)
        start = time.perf_counter_ns()
        func(valid, done)
        end = time.perf_counter_ns()
        times[n] = (end - start) / 1000

    print(f"Average time = {times.mean():.3f} +/- {times.std():.3f} us")
    print()


if __name__ == "__main__":
    shape = (128, 32)
    n_repeat = 1000
    rng = random.default_rng()

    funcs = [naive, using_output_arg, no_intermediates]
    names = [func.__name__ for func in funcs]

    funcs += [njit(naive)]
    names += ["naive+njit"]

    for func, name in zip(funcs, names):
        print(f"Benchmarking {name}")
        benchmark(func, rng, shape, n_repeat)
