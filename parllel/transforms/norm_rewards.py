from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numba import njit

from parllel import Array, ArrayDict

from .running_mean_std import RunningMeanStd
from .transform import Transform

EPSILON = 1e-6


@njit(fastmath=True)
def compute_past_discount_return(
    reward: np.ndarray,
    done: np.ndarray,
    previous_return: np.ndarray,
    previous_done: np.ndarray,
    discount: float,
    out_return: np.ndarray,
) -> None:
    """Computes discounted sum of past rewards from T=0 to the current time
    step. Rewards received further in the past are more heavily discounted.
    """
    return_ = out_return
    not_done = np.logical_not(done)
    previous_not_done = np.logical_not(previous_done)
    return_[0] = reward[0] + previous_return * discount * previous_not_done
    for t in range(1, len(reward)):
        return_[t] = reward[t] + return_[t - 1] * discount * not_done[t - 1]


class NormalizeRewards(Transform):
    """Normalizes rewards by dividing by the standard deviation of past
    discounted returns. As a side-effect, adds past_return to the samples
    buffer for the discounted returns gained by the agent up to the current
    time.

    If valid exists, then only advantage of steps where valid == True
    are used for calculating statistics. Other data points are ignored.

    Requires fields:
        - reward
        - done
        - [valid]

    Requires output fields:
        - past_return

    :param sample_tree: the ArrayDict that will be passed to `__call__`.
    :param discount: discount (gamma) for discounting rewards over time
    :param initial_count: seed the running mean and standard deviation model
        with `initial_count` instances of x~N(0,1). Increase this to improve
        stability, to prevent the mean and standard deviation from changing too
        quickly during early training.
    """

    def __init__(
        self,
        sample_tree: ArrayDict[Array],
        discount: float,
        initial_count: float | None = None,
    ) -> None:
        if isinstance(sample_tree["reward"], Mapping):
            raise NotImplementedError(
                "Not implemented for markov games, where" "rewards are agent-specific."
            )

        self.only_valid = "valid" in sample_tree
        self.discount = discount

        if initial_count is not None and initial_count < 1.0:
            raise ValueError("Initial count must be at least 1")

        # create model to track running mean and std_dev of samples
        if initial_count is not None:
            self.return_statistics = RunningMeanStd(
                shape=(), initial_count=initial_count
            )
        else:
            self.return_statistics = RunningMeanStd(shape=())

    def __call__(self, sample_tree: ArrayDict[Array]) -> ArrayDict[Array]:
        reward = sample_tree["reward"]
        assert isinstance(reward, Array)
        assert len(reward.batch_shape) in (1, 2)
        reward = np.asarray(reward)
        past_return = sample_tree["past_return"]
        previous_past_return = np.asarray(past_return[-1])
        past_return = np.asarray(past_return)
        done = sample_tree["done"]
        previous_done = np.asarray(done[-1])
        done = np.asarray(done)

        compute_past_discount_return(
            reward=reward,
            done=done,
            previous_return=previous_past_return,
            previous_done=previous_done,
            discount=self.discount,
            out_return=past_return,
        )

        # update statistics of discounted return
        if self.only_valid:
            valid = sample_tree["valid"]
            self.return_statistics.update(past_return[valid])
        else:
            self.return_statistics.update(past_return)

        reward[:] = reward / (np.sqrt(self.return_statistics.var + EPSILON))

        return sample_tree
