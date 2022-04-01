import time

from numba import njit
import numpy as np
from numpy import random

mean = 5
std_dev = 3
EPSILON = 1e-6

def generate(rng, shape):
    return rng.normal(mean, std_dev, size=shape)

def naive(obs):
    for t in range(obs.shape[0]):
        obs[t] = (obs[t] - mean) / (std_dev + EPSILON)

def naive_in_two_steps(obs):
    for t in range(obs.shape[0]):
        obs[t] = (obs[t] - mean)
        obs[t] = obs[t] / (std_dev + EPSILON)

def multiply_with_out_arg(obs):
    for t in range(obs.shape[0]):
        np.multiply(obs[t] - mean, 1 / (std_dev + EPSILON), out=obs[t])

def divide_with_out_arg(obs):
    for t in range(obs.shape[0]):
        np.divide(obs[t] - mean, (std_dev + EPSILON), out=obs[t])

def subtract_multiply_with_out_arg(obs):
    for t in range(obs.shape[0]):
        np.subtract(obs[t], mean, out=obs[t])
        np.multiply(obs[t], 1 / (std_dev + EPSILON), out=obs[t])

def subtract_divide_with_out_arg(obs):
    for t in range(obs.shape[0]):
        np.subtract(obs[t], mean, out=obs[t])
        np.divide(obs[t], std_dev + EPSILON, out=obs[t])

def no_for_loop(obs):
    obs[:] = (obs - mean) / (std_dev + EPSILON)


def benchmark(func, rng, shape, n_repeat):
    obs = generate(rng, shape)

    # first run not timed, in case we need to jit
    func(obs)

    times = np.zeros(n_repeat, dtype=float)

    for n in range(n_repeat):
        obs = generate(rng, shape)
        start = time.perf_counter_ns()
        func(obs)
        end = time.perf_counter_ns()
        times[n] = (end - start) / 1000

    print(f"Average time = {times.mean():.3f} +/- {times.std():.3f} us")
    print()


if __name__ == "__main__":
    shape = (128, 32)
    n_repeat = 1000
    rng = random.default_rng()

    funcs = [naive, naive_in_two_steps, multiply_with_out_arg, divide_with_out_arg,
        subtract_multiply_with_out_arg, subtract_divide_with_out_arg]
    names = [func.__name__ for func in funcs]

    funcs += [njit(naive), njit(naive_in_two_steps), njit(no_for_loop)]
    names += ["naive+njit", "naive_in_two_steps+njit", "no_for_loop+njit"]

    for func, name in zip(funcs, names):
        print(f"Benchmarking {name}")
        benchmark(func, rng, shape, n_repeat)
