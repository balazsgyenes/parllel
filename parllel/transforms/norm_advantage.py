import numpy as np
from nptyping import NDArray

from parllel.buffers import Samples

from .transform import BatchTransform


EPSILON = 1e-6


class NormalizeAdvantage(BatchTransform):
    """Batch normalizes advantage by subtracting the mean and dividing by the
    standard deviation of the current batch of advantage values. If
     exists, then only advantage of valid steps are
    included in the statistics.

    Requires fields:
        - .env.advantage
        - [.env.valid]

    :param only_valid: when calculating statistics, only use data points where
        `batch_samples.env.valid` is True. Other data points are ignored. This
        should be True if mid-batch resets are turned off.
    """
    def __init__(self, only_valid: bool) -> None:
        self.only_valid = only_valid

    def __call__(self, batch_samples: Samples) -> Samples:
        advantage = np.asarray(batch_samples.env.advantage)

        # update statistics of discounted return
        if self.only_valid:
            valid = batch_samples.env.valid
            valid_advantage: NDArray = advantage[valid]
            mean = valid_advantage.mean()
            std = valid_advantage.std()
        else:
            mean = advantage.mean()
            std = advantage.std()

        advantage[:] = (advantage - mean) / (std + EPSILON)

        return batch_samples
