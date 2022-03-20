from numba import njit
import numpy as np
from nptyping import NDArray

from parllel.buffers import NamedArrayTupleClass
from parllel.samplers import Samples, EnvSamples

from .transform import Transform


@njit
def valid_from_done(done: NDArray[np.bool_], out_valid: NDArray[np.bool_]):
    """Returns a float mask which is zero for all time-steps after a
    `done=True` is signaled.  This function operates on the leading dimension
    of `done`, assumed to correspond to time [T,...], other dimensions are
    preserved."""
    valid = out_valid
    valid[0] = True # first time step is always valid
    not_done = np.logical_not(done)
    for t in range(1, len(valid)):
        # valid if previous step was valid and not done yet
        valid[t] = np.logical_and(valid[t-1], not_done[t-1])
    return valid


class ValidFromDone(Transform):
    def __init__(self) -> None:
        self._EnvSamplesClass = None
        
    def dry_run(self, batch_samples: Samples) -> Samples:
        env_samples: EnvSamples = batch_samples.env

        EnvSamplesClass = NamedArrayTupleClass(
            typename = env_samples._typename,
            fields = env_samples._fields + ("valid",)
        )

        valid = np.zeros_like(batch_samples.env.reward)

        valid_from_done(
            batch_samples.env.done,
            valid,
        )

        env_samples = EnvSamplesClass(
            **env_samples._asdict(), valid=valid,
        )

        batch_samples = Samples(env=env_samples, agent=batch_samples.agent)
        return batch_samples

    def __call__(self, batch_samples: Samples) -> Samples:
        valid_from_done(
            batch_samples.env.done,
            batch_samples.env.valid,
        )
        return batch_samples
