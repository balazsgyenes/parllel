from numba import njit
import numpy as np
from nptyping import NDArray

from parllel.buffers import Samples

from .transform import BatchTransform


@njit(fastmath=True)
def compute_discount_return(
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
    advantage[...] = return_ - value


@njit(fastmath=True)
def compute_gae_advantage(
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
    return_[...] = advantage + value


class EstimateAdvantage(BatchTransform):
    def __init__(self, discount: float, gae_lambda: float) -> None:
        """Adds a field to samples buffer under `env.advantage` for the
        advantage: roughly, the return left to go compared to the state value
        predicted by the agent. The agent's bootstrap value accounts for
        rewards that are expected to be gained after the end of the current
        batch. If `lambda==1.0`, advantage is estimated as discounted return
        minus the value, otherwise Generalized Advantage Estimation (GAE) is\
        used.
        
        Requires fields:
            - .env.reward
            - .env.done
            - .agent.agent_info.value
            - .agent.bootstrap_value
        
        Adds fields:
            - .env.advantage
            - .env.return_

        :param discount: discount (gamma) for discounting rewards over time
        :param gae_lambda: lambda parameter for GAE algorithm
        """
        self.discount = discount
        self.gae_lambda = gae_lambda
        if gae_lambda == 1.0:
            # GAE reduces to empirical discounted return
            self.estimator = compute_discount_return
        else:
            self.estimator = compute_gae_advantage

    def __call__(self, batch_samples: Samples) -> Samples:
        self.estimator(
            np.asarray(batch_samples.env.reward),
            np.asarray(batch_samples.agent.agent_info.value),
            np.asarray(batch_samples.env.done),
            np.asarray(batch_samples.agent.bootstrap_value),
            self.discount,
            self.gae_lambda,
            np.asarray(batch_samples.env.advantage),
            np.asarray(batch_samples.env.return_),
        )
        return batch_samples
