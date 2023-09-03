from __future__ import annotations

import numpy as np
from numba import njit

from parllel import Array, ArrayDict

from .transform import Transform


@njit(fastmath=True)
def compute_discount_return(
    reward: np.ndarray,
    value: np.ndarray,
    done: np.ndarray,
    bootstrap_value: np.ndarray,
    discount: float,
    gae_lambda: float,  # keep this here so the signatures match
    out_advantage: np.ndarray,
    out_return: np.ndarray,
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
    reward: np.ndarray,
    value: np.ndarray,
    done: np.ndarray,
    bootstrap_value: np.ndarray,
    discount: float,
    gae_lambda: float,
    out_advantage: np.ndarray,
    out_return: np.ndarray,
) -> None:
    """Time-major inputs, optional other dimensions: [T], [T,B], etc.  Similar
    to `discount_return()` but using Generalized Advantage Estimation to
    compute advantages and returns."""
    advantage, return_ = out_advantage, out_return
    not_done = np.logical_not(done)
    advantage[-1] = reward[-1] + discount * bootstrap_value * not_done[-1] - value[-1]
    # for t in reversed(range(len(reward) - 1)): # numba doesn't support reversed
    for t in range(len(reward) - 2, -1, -1):  # iterate backwards through time
        delta = reward[t] + discount * value[t + 1] * not_done[t] - value[t]
        advantage[t] = delta + discount * gae_lambda * not_done[t] * advantage[t + 1]
    return_[...] = advantage + value


class EstimateAdvantage(Transform):
    """Adds an estimate of the advantage function for each sample under
    `advantage`. Advantage is estimated either using the empirical
    discounted return minus the agent's value estimate (if `gae_lambda==1.0`)
    or using Generalized Advantage Estimation (if `gae_lambda<1.0`). The
    agent's value estimate at the end of a batch accounts for returns left to
    go in the trajectory (called a bootstrap value).

    Requires fields:
        - reward
        - done
        - agent_info.value
        - bootstrap_value

    Requires output fields:
        - advantage
        - return_

    :param discount: discount (gamma) for discounting rewards over time
    :param gae_lambda: lambda parameter for GAE algorithm
    """

    def __init__(self, discount: float, gae_lambda: float) -> None:
        self.discount = discount
        self.gae_lambda = gae_lambda
        if gae_lambda == 1.0:
            # GAE reduces to empirical discounted return
            self.estimator = compute_discount_return
        else:
            self.estimator = compute_gae_advantage

    def __call__(self, sample_tree: ArrayDict[Array]) -> ArrayDict[Array]:
        advantage = sample_tree["advantage"]
        assert isinstance(advantage, Array)
        assert len(advantage.batch_shape) == 2

        self.estimator(
            np.asarray(sample_tree["reward"]),
            np.asarray(sample_tree["agent_info"]["value"]),
            np.asarray(sample_tree["done"]),
            np.asarray(sample_tree["bootstrap_value"]),
            self.discount,
            self.gae_lambda,
            np.asarray(advantage),
            np.asarray(sample_tree["return_"]),
        )
        return sample_tree
