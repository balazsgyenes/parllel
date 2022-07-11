from typing import Optional

import numpy as np
from nptyping import NDArray
from numba import njit

from parllel.arrays import Array, RotatingArray
from parllel.buffers import EnvSamples, NamedArrayTupleClass, Samples

from .running_mean_std import RunningMeanStd
from .transform import BatchTransform


EPSILON = 1e-6


@njit(fastmath=True)
def compute_past_discount_return(
        reward: NDArray[np.float32],
        done: NDArray[np.bool_],
        previous_return: NDArray[np.float32],
        previous_done: NDArray[np.bool_],
        discount: float,
        out_return: NDArray[np.float32],
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


class NormalizeRewards(BatchTransform):
    def __init__(self,
        discount: float,
        only_valid: bool,
        initial_count: Optional[float] = None
    ) -> None:
        """Normalizes rewards by dividing by the standard deviation of past
        discounted returns. As a side-effect, adds past_return to the samples
        buffer for the discounted returns gained by the agent up to the current
        time.

        Requires fields:
            - .env.reward
            - .env.done

        Adds fields:
            - .env.past_return

        :param discount: discount (gamma) for discounting rewards over time
        :param initial_count: seed the running mean and standard deviation
            model with `initial_count` instances of x~N(0,1). Increase this to
            improve stability, to prevent the mean and standard deviation from
            changing too quickly during early training
        """
        self.discount = discount
        self.only_valid = only_valid

        if initial_count is not None and initial_count < 1.:
            raise ValueError("Initial count must be at least 1")

        # create model to track running mean and std_dev of samples
        if initial_count is not None:
            self.return_statistics = RunningMeanStd(shape=(),
                initial_count=initial_count)
        else:
            self.return_statistics = RunningMeanStd(shape=())

    def __call__(self, batch_samples: Samples) -> Samples:
        reward = np.asarray(batch_samples.env.reward)
        past_return = batch_samples.env.past_return
        previous_past_return = np.asarray(past_return[past_return.first - 1])
        past_return = np.asarray(past_return)
        done = batch_samples.env.done
        previous_done = np.asarray(done[done.first - 1])
        done = np.asarray(done)

        compute_past_discount_return(
            reward,
            done,
            previous_past_return,
            previous_done,
            self.discount,
            past_return,
        )

        # update statistics of discounted return
        if self.only_valid:
            valid = batch_samples.env.valid
            self.return_statistics.update(past_return[valid])
        else:
            self.return_statistics.update(past_return)

        reward[:] = reward / (np.sqrt(self.return_statistics.var + EPSILON))

        return batch_samples
