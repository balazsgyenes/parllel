from __future__ import annotations

from nptyping import NDArray
from numba import njit
import numpy as np


@njit
def update_from_moments(
    batch_mean: NDArray,
    batch_var: NDArray,
    batch_count: int,
    mean: NDArray,
    var: NDArray,
    count: NDArray,
) -> None:
    """Calculates an update to a running mean, std_dev, and count based from
    the mean, std_dev, and count of an incoming batch.

    Adapted from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
    Adapted from https://github.com/astooke/rlpyt/blob/master/rlpyt/models/running_mean_std.py 
    References https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    delta = batch_mean - mean
    total_count = batch_count + count

    mean[...] = mean + delta * batch_count / total_count
    m_a = var * count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + np.square(delta) * count * batch_count / total_count
    var[...] = m_2 / total_count

    count[...] = total_count


class RunningMeanStd:
    """
    Calculates the running mean and std of a data stream

    :param shape: the shape of the data stream's output
    :param initial_count: helps with arithmetic issues and stability
    """
    def __init__(self, shape: tuple[int, ...], initial_count: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = np.array(initial_count, np.float64)
        self.shape = shape

    def update(self, batch: np.ndarray) -> None:
        batch = batch.reshape(-1, *self.shape)

        batch_mean = np.mean(batch, 0)
        batch_var = np.var(batch, 0)
        batch_count = batch.shape[0]
        update_from_moments(batch_mean, batch_var, batch_count,
                            self.mean, self.var, self.count)
