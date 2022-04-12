from typing import Optional

import numpy as np

from parllel.buffers import Samples

from .running_mean_std import RunningMeanStd
from .transform import StepTransform


EPSILON = 1e-6


class NormalizeObservations(StepTransform):
    def __init__(self, initial_count: Optional[float] = None) -> None:
        """Normalizes the observation by subtracting the mean and dividing by
        the standard deviation.

        :param initial_count: seed the running mean and standard deviation
            model with `initial_count` instances of x~N(0,1). Increase this to
            improve stability, to prevent the mean and standard deviation from
            changing too quickly during early training
        """
        if initial_count is not None and initial_count < 1.:
            raise ValueError("Initial count must be at least 1")
        self._initial_count = initial_count

    def dry_run(self, batch_samples: Samples) -> Samples:
        # get shape of observation assuming 2 leading dimensions
        obs_shape = batch_samples.env.observation.shape[2:]

        self.only_valid = True if hasattr(batch_samples.env, "valid") else False

        # create model to track running mean and std_dev of samples
        if self._initial_count is not None:
            self._obs_statistics = RunningMeanStd(shape=obs_shape,
                initial_count=self._initial_count)
        else:
            self._obs_statistics = RunningMeanStd(shape=obs_shape)
        
        return batch_samples

    def __call__(self, batch_samples: Samples, t: int) -> Samples:
        step_obs = np.asarray(batch_samples.env.observation[t])

        # update statistics of each element of observation
        if self.only_valid:
            valid = batch_samples.env.valid[t]
            # this fancy indexing operation creates a copy, but that's fine
            self._obs_statistics.update(step_obs[valid])
        else:
            self._obs_statistics.update(step_obs)

        step_obs[:] = (step_obs - self._obs_statistics.mean) / (
            np.sqrt(self._obs_statistics.var + EPSILON))

        return batch_samples
