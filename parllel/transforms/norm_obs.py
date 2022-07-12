from typing import Optional, Tuple

import numpy as np

from parllel.buffers import Samples

from .running_mean_std import RunningMeanStd
from .transform import StepTransform


EPSILON = 1e-6


class NormalizeObservations(StepTransform):
    """Normalizes the observation by subtracting the mean and dividing by the
    standard deviation.

    Requires fields:
        - .env.observation
        - [.env.valid]

    :param obs_shape: shape of a single observation
    :param only_valid: when calculating statistics, only use data points where
        `batch_samples.env.valid` is True. Other data points are ignored. This
        should be True if mid-batch resets are turned off.
    :param initial_count: seed the running mean and standard deviation model
        with `initial_count` instances of x~N(0,1). Increase this to improve
        stability, to prevent the mean and standard deviation from changing too
        quickly during early training.
    """
    def __init__(self,
            obs_shape: Tuple[int, ...],
            only_valid: bool,
            initial_count: Optional[float] = None,
        ) -> None:
        self.only_valid = only_valid

        if initial_count is not None and initial_count < 1.:
            raise ValueError("Initial count must be at least 1")

        # create model to track running mean and std_dev of samples
        if initial_count is not None:
            self.obs_statistics = RunningMeanStd(shape=obs_shape,
                initial_count=initial_count)
        else:
            self.obs_statistics = RunningMeanStd(shape=obs_shape)

    def __call__(self, batch_samples: Samples, t: int) -> Samples:
        step_obs = np.asarray(batch_samples.env.observation[t])

        # update statistics of each element of observation
        if self.only_valid:
            valid = batch_samples.env.valid[t]
            # this fancy indexing operation creates a copy, but that's fine
            self.obs_statistics.update(step_obs[valid])
        else:
            self.obs_statistics.update(step_obs)

        step_obs[:] = (step_obs - self.obs_statistics.mean) / (
            np.sqrt(self.obs_statistics.var + EPSILON))

        return batch_samples
