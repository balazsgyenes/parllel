from typing import Optional

import numpy as np

from parllel.samplers import Samples

from .running_mean_std import RunningMeanStd
from .transform import Transform


EPSILON = 1e-6


class NormalizeObservations(Transform):
    def __init__(self, initial_count: Optional[float] = None) -> None:
        """Normalizes the observation after every time step. This transform can
        only be used as a step transform.

        :param initial_count: seed the running mean and standard deviation
            model with `initial_count` instances of x~N(0,1). Increase this to
            improve stability, to prevent the mean and standard deviation from
            changing too quickly during early training
        """
        if initial_count is not None and initial_count < 1.:
            raise ValueError("Initial must be at least 1")
        self._initial_count = initial_count

    def dry_run(self, batch_samples: Samples) -> Samples:
        # get shape of observation assuming 2 leading dimensions
        obs_shape = batch_samples.env.observation.shape[2:]

        # create model to track running mean and std_dev of samples
        if self._initial_count is not None:
            self._obs_statistics = RunningMeanStd(shape=obs_shape,
                initial_count=self._initial_count)
        else:
            self._obs_statistics = RunningMeanStd(shape=obs_shape)
        
        return batch_samples

    def __call__(self, batch_samples: Samples, t: Optional[int] = None) -> Samples:
        step_obs = np.asarray(batch_samples.env.observation[t])

        # update statistics of each element of observation
        self._obs_statistics.update(step_obs)

        np.subtract(step_obs, self._obs_statistics.mean, out=step_obs)
        np.multiply(step_obs, 1 / (np.sqrt(self._obs_statistics.var + EPSILON)),
            out=step_obs)

        return batch_samples
