from __future__ import annotations

from nptyping import NDArray
import numpy as np

from parllel import Array, ArrayDict

from .transform import BatchTransform


EPSILON = 1e-6


class NormalizeAdvantage(BatchTransform):
    """Batch normalizes advantage by subtracting the mean and dividing by the
    standard deviation of the current batch of advantage values.
    
    If .env.valid exists, then only advantage of steps where .env.valid == True
    are used for calculating statistics. Other data points are ignored.
    
    Requires fields:
        - .env.advantage
        - [.env.valid]

    :param sample_tree: the ArrayDict that will be passed to `__call__`.
    """
    def __init__(self, sample_tree: ArrayDict[Array]) -> None:
        self.only_valid = "valid" in sample_tree
        self.multiagent = np.asarray(sample_tree["advantage"]).ndim > 2

    def __call__(self, sample_tree: ArrayDict[Array]) -> ArrayDict[Array]:
        advantage = np.asarray(sample_tree["advantage"])

        # calculate batch mean and stddev, optionally considering only valid
        valid_advantage = advantage
        if self.only_valid:
            valid = sample_tree["valid"]
            # shape is [X] for single-agent case, and [X, N] for multiagent
            # where X is number of valid time steps and N is number of agents
            valid_advantage: NDArray = valid_advantage[valid]

        if self.multiagent:
            # normalize over all but last axis
            axes = tuple(range(valid_advantage.ndim - 1))
        else:
            # normalize over all axes
            axes = None

        mean = valid_advantage.mean(axis=axes)
        std = valid_advantage.std(axis=axes)

        advantage[...] = (advantage - mean) / (std + EPSILON)

        return sample_tree
