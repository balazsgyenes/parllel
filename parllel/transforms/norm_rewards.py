from typing import Optional

import numpy as np
from nptyping import NDArray
from numba import njit

from parllel.arrays import Array, RotatingArray
from parllel.buffers import NamedArrayTupleClass
from parllel.samplers import Samples, EnvSamples

from .running_mean_std import RunningMeanStd
from .transform import Transform


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


class NormalizeRewards(Transform):
    def __init__(self,
        discount: float,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
    ) -> None:
        self._discount = discount
        self._reward_min = reward_min
        self._reward_max = reward_max
        self._do_clip = reward_min is not None or reward_max is not None
        self._reward_statistics = RunningMeanStd(shape=(1,))
    
    def dry_run(self, batch_samples: Samples, RotatingArrayCls: Array) -> Samples:
        # get convenient local references
        env_samples: EnvSamples = batch_samples.env
        reward = env_samples.reward
        done = env_samples.done

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

        # if done is not a rotating array, reallocate as a rotating array
        if not isinstance(env_samples.done, RotatingArray):
            done = RotatingArrayCls(shape=done.shape, dtype=done.dtype, padding=1)
            done[:] = env_samples.done
            env_samples_dict["done"] = done

        # package everything back into batch_samples
        env_samples = EnvSamplesClass(
            **env_samples_dict, past_return=past_return,
        )
        batch_samples = batch_samples._replace(env=env_samples)

        # test the forward pass
        self.__call__(batch_samples)

        return batch_samples

    def __call__(self, batch_samples: Samples) -> Samples:

        reward = np.asarray(batch_samples.env.reward)
        done = np.asarray(batch_samples.env.done)
        past_return = np.asarray(batch_samples.env.past_return)
        previous_past_return = np.asarray(batch_samples.env.past_return[-1])
        previous_done = np.asarray(batch_samples.env.done[-1])

        # TODO: modify this function to take just 4 arguments, arrays of different lengths
        compute_past_discount_return(
            reward,
            done,
            previous_past_return,
            previous_done,
            self._discount,
            past_return,
            )

        # update statistics of discounted return
        self._reward_statistics.update(past_return)

        np.multiply(reward, 1 / (np.sqrt(self._reward_statistics.var + EPSILON)),
            out=reward)

        if self._do_clip:
            np.clip(reward, self._reward_min, self._reward_max, out=reward)
        
        return batch_samples
