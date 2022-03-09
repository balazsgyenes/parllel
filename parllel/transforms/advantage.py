
from numba import njit
import numpy as np
from nptyping import NDArray

from parllel.buffers import Buffer, NamedArrayTupleClass
from parllel.samplers import Samples, EnvSamples
from parllel.transforms import Transform


@njit
def generalized_advantage_estimation(
        reward: Buffer[NDArray[np.float32]],
        value: Buffer[NDArray[np.float32]],
        done: Buffer[NDArray[np.bool_]],
        bootstrap_value: Buffer[NDArray[np.float32]],
        discount: float,
        gae_lambda: float,
        out_advantage: Buffer[NDArray[np.float32]],
        out_return: Buffer[NDArray[np.float32]],
    ):
    """Time-major inputs, optional other dimensions: [T], [T,B], etc.  Similar
    to `discount_return()` but using Generalized Advantage Estimation to
    compute advantages and returns."""
    advantage, return_ = out_advantage, out_return
    not_done = 1 - done
    not_done = not_done.astype(reward.dtype)
    advantage[-1] = reward[-1] + discount * bootstrap_value * not_done[-1] - value[-1]
    # reversed(range(len(reward) - 1)), but numba doesn't support reversed
    for t in range(len(reward) - 2, -1, -1): # iterate backwards through time
        delta = reward[t] + discount * value[t + 1] * not_done[t] - value[t]
        advantage[t] = delta + discount * gae_lambda * not_done[t] * advantage[t + 1]
    return_[:] = advantage + value


class GeneralizedAdvantageEstimator(Transform):
    def __init__(self, discount: float, gae_lambda: float) -> None:
        self._discount = discount
        self._lambda = gae_lambda

        self._EnvSamplesClass = None

    def __call__(self, batch_samples: Samples) -> Buffer:

        env_samples: EnvSamples = batch_samples.env

        if self._EnvSamplesClass is None:
            self._EnvSamplesClass = NamedArrayTupleClass(
                typename = env_samples._typename + "WithAdvantage",
                fields = env_samples._fields + ("advantage", "return_")
            )

        # TODO: pre-allocate these as part of overall samples buffer
        advantage = np.zeros_like(batch_samples.env.reward)
        return_ = np.zeros_like(batch_samples.env.reward)

        generalized_advantage_estimation(
            batch_samples.env.reward,
            batch_samples.agent.agent_info.value,
            batch_samples.env.done,
            batch_samples.agent.bootstrap_value,
            self._discount,
            self._lambda,
            advantage,
            return_,
        )

        env_samples = self._EnvSamplesClass(
            **env_samples._asdict(), advantage=advantage, return_=return_,
        )

        batch_samples = Samples(env=env_samples, agent=batch_samples.agent)
        return batch_samples
