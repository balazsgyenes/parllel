from __future__ import annotations

import numpy as np

from parllel import Array, ArrayDict

from .transform import Transform


class ClipRewards(Transform):
    """Clips rewards between maximum and minimum values. If either bound is not
    given, rewards are not clipped from that direction.

    Requires fields:
        - reward

    :param reward_min: after normalization, clips rewards from below
    :param reward_max: after normalization, clips rewards from above
    """

    def __init__(
        self,
        reward_min: float | None = None,
        reward_max: float | None = None,
    ) -> None:
        self.reward_min = reward_min
        self.reward_max = reward_max
        if not (reward_min is not None or reward_max is not None):
            raise ValueError("Must provide either reward_min or reward_max")

    def __call__(self, sample_tree: ArrayDict[Array]) -> ArrayDict[Array]:
        reward = sample_tree["reward"]
        assert isinstance(reward, Array)
        reward_np = np.asarray(reward)

        reward[:] = np.clip(reward_np, self.reward_min, self.reward_max)

        return sample_tree
