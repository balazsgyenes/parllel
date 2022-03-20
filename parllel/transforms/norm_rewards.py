import numpy as np
from nptyping import NDArray
from numba import njit

from parllel.samplers import Samples

from .running_mean_std import RunningMeanStd
from .transform import Transform


EPSILON = 1e-6


@njit(fastmath=True)
def past_discount_return(
        reward: NDArray[np.float32],
        done: NDArray[np.bool_],
        discount: float,
        out_return: NDArray[np.float32],
    ) -> None:
    """Computes discounted sum of past rewards from T=0 to the current time
    step. Rewards received further in the past are more heavily discounted.
    """
    return_ = out_return
    not_done = np.logical_not(done)
    return_[0] = reward[0]
    for t in range(1, len(reward)):
        return_[t] = reward[t] + return_[t - 1] * discount * not_done[t - 1]
    return return_


class NormalizeRewards(Transform):
    def __init__(self, discount: float) -> None:
        self._discount = discount
        self._reward_statistics = RunningMeanStd()
    
    def dry_run(self, batch_samples: Samples) -> Samples:
        # TODO: allocate past_disc_reward as a RotatingArray
        # last discounted return of previous batch as starting point for next
        # batch
        raise NotImplementedError

    def __call__(self, batch_samples: Samples) -> Samples:

        discounted_returns = past_discount_return(
            batch_samples.env.reward,
            batch_samples.env.done,
            self._reward,
            self.past_disc_reward,
            )

        reward = batch_samples.env.reward
        np.multiply(reward, 1 / (np.sqrt(self.reward_statistics.var + EPSILON)),
            out=reward)
