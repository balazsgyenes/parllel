from numba import njit
import numpy as np
from nptyping import NDArray

from parllel.buffers import NamedArrayTupleClass
from parllel.samplers import Samples, EnvSamples

from .transform import Transform


@njit(fastmath=True)
def discount_return(
        reward: NDArray[np.float32],
        value: NDArray[np.float32],
        done: NDArray[np.bool_],
        bootstrap_value: NDArray[np.float32],
        discount: float,
        gae_lambda: float,  # keep this here so the signatures match
        out_advantage: NDArray[np.float32],
        out_return: NDArray[np.float32],
    ) -> None:
    """Time-major inputs, optional other dimensions: [T], [T,B], etc. Computes
    discounted sum of future rewards from each time-step to the end of the
    batch, including bootstrapping value.  Sum resets where `done` is 1.
    Optionally, writes to buffer `return_dest`, if provided.  Operations
    vectorized across all trailing dimensions after the first [T,]."""
    advantage, return_ = out_advantage, out_return
    not_done = np.logical_not(done)
    return_[-1] = reward[-1] + discount * bootstrap_value * not_done[-1]
    # for t in reversed(range(len(reward) - 1)): # numba doesn't support reversed
    for t in range(len(reward) - 2, -1, -1):
        return_[t] = reward[t] + return_[t + 1] * discount * not_done[t]
    advantage[:] = return_ - value


@njit(fastmath=True)
def generalized_advantage_estimation(
        reward: NDArray[np.float32],
        value: NDArray[np.float32],
        done: NDArray[np.bool_],
        bootstrap_value: NDArray[np.float32],
        discount: float,
        gae_lambda: float,
        out_advantage: NDArray[np.float32],
        out_return: NDArray[np.float32],
    ) -> None:
    """Time-major inputs, optional other dimensions: [T], [T,B], etc.  Similar
    to `discount_return()` but using Generalized Advantage Estimation to
    compute advantages and returns."""
    advantage, return_ = out_advantage, out_return
    not_done = np.logical_not(done)
    advantage[-1] = reward[-1] + discount * bootstrap_value * not_done[-1] - value[-1]
    # for t in reversed(range(len(reward) - 1)): # numba doesn't support reversed
    for t in range(len(reward) - 2, -1, -1): # iterate backwards through time
        delta = reward[t] + discount * value[t + 1] * not_done[t] - value[t]
        advantage[t] = delta + discount * gae_lambda * not_done[t] * advantage[t + 1]
    return_[:] = advantage + value


class GeneralizedAdvantageEstimator(Transform):
    def __init__(self, discount: float, gae_lambda: float) -> None:
        self._discount = discount
        self._lambda = gae_lambda
        if gae_lambda == 1.0:
            # GAE reduces to empirical discounted return
            self.estimator = discount_return
        else:
            self.estimator = generalized_advantage_estimation

        self._EnvSamplesClass = None

    def __call__(self, batch_samples: Samples) -> Samples:

        env_samples: EnvSamples = batch_samples.env

        if self._EnvSamplesClass is None:
            self._EnvSamplesClass = NamedArrayTupleClass(
                typename = env_samples._typename,
                fields = env_samples._fields + ("advantage", "return_")
            )

        # TODO: pre-allocate these as part of overall samples buffer
        advantage = np.zeros_like(batch_samples.env.reward)
        return_ = np.zeros_like(batch_samples.env.reward)

        self.estimator(
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
