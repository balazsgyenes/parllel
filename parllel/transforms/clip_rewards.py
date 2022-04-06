from typing import Optional

import numpy as np

from parllel.buffers import Samples

from .transform import BatchTransform


class ClipRewards(BatchTransform):
    def __init__(self,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
    ) -> None:
        """Clips rewards between maximum and minimum values. If either bound is
        not given, rewards are not clipped from that direction.

        Requires fields:
            - .env.reward
            - .env.done

        :param reward_min: after normalization, clips rewards from below
        :param reward_max: after normalization, clips rewards from above
        """
        self._reward_min = reward_min
        self._reward_max = reward_max
        if not (reward_min is not None or reward_max is not None):
            raise ValueError("Must provide either reward_min or reward_max")
    
    def dry_run(self, batch_samples: Samples) -> Samples:
        return batch_samples

    def __call__(self, batch_samples: Samples) -> Samples:
        reward = np.asarray(batch_samples.env.reward)

        reward[:] = np.clip(reward, self._reward_min, self._reward_max)
        
        return batch_samples
