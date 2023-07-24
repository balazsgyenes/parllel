from __future__ import annotations

import numpy as np

from parllel import Array, ArrayDict

from .transform import BatchTransform


class ClipRewards(BatchTransform):
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

    def __call__(self, batch_samples: ArrayDict[Array]) -> ArrayDict[Array]:
        reward = np.asarray(batch_samples["reward"])

        reward[:] = np.clip(reward, self.reward_min, self.reward_max)

        return batch_samples
