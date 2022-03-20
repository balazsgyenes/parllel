from typing import Tuple

import numpy as np
from nptyping import NDArray
from numba import njit


@njit
def update_from_moments(
    batch_mean: NDArray,
    batch_var: NDArray,
    batch_count: int,
    mean: NDArray,
    var: NDArray,
    count: NDArray,
) -> None:
    delta = batch_mean - mean
    total_count = batch_count + count

    mean[:] = mean + delta * batch_count / total_count
    m_a = var * count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + np.square(delta) * count * batch_count / total_count
    var[:] = m_2 / total_count

    count[:] = total_count


class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param shape: the shape of the data stream's output
        :param epsilon: helps with arithmetic issues
        """
        assert len(shape) > 0
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = np.array([epsilon], np.float64)
        self.shape = shape

    def update(self, batch: np.ndarray) -> None:
        batch = batch.reshape(-1, *self.shape)

        batch_mean = np.mean(batch, 0)
        batch_var = np.var(batch, 0)
        batch_count = batch.shape[0]
        update_from_moments(batch_mean, batch_var, batch_count,
                            self.mean, self.var, self.count)
