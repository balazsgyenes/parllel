from typing import Optional

import numpy as np
from nptyping import NDArray
from numba import njit

from parllel.arrays import Array, RotatingArray
from parllel.buffers import NamedArrayTupleClass
from parllel.samplers import Samples, EnvSamples

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
        self._discount = discount
        if initial_count is not None and initial_count < 1.:
            raise ValueError("Initial must be at least 1")
        self._initial_count = initial_count
    
    def dry_run(self, batch_samples: Samples, RotatingArrayCls: Array) -> Samples:
        # get convenient local references
        env_samples: EnvSamples = batch_samples.env
        reward = env_samples.reward

        if not isinstance(env_samples.done, RotatingArray):
            raise TypeError("batch_samples.env.done must be a RotatingArray "
                            "when using NormalizeRewards")

        # create new NamedArrayTuple for env samples with additional field
        EnvSamplesClass = NamedArrayTupleClass(
            typename = env_samples._typename,
            fields = env_samples._fields + ("past_return",)
        )
        # get contents of old env samples as a dictionary
        env_samples_dict = env_samples._asdict()

        # allocate new Array for past discounted returns
        past_return = RotatingArrayCls(shape=reward.shape,
            dtype=reward.dtype, padding=1)

        # package everything back into batch_samples
        env_samples = EnvSamplesClass(
            **env_samples_dict, past_return=past_return,
        )
        batch_samples = batch_samples._replace(env=env_samples)

        # create model to track running mean and std_dev of samples
        if self._initial_count is not None:
            self._return_statistics = RunningMeanStd(shape=(),
                initial_count=self._initial_count)
        else:
            self._return_statistics = RunningMeanStd(shape=())

        return batch_samples

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
            self._discount,
            past_return,
            )

        # update statistics of discounted return
        self._return_statistics.update(past_return)

        np.multiply(reward, 1 / (np.sqrt(self._return_statistics.var + EPSILON)),
            out=reward)

        return batch_samples
