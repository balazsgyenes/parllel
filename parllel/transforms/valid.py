from numba import njit
import numpy as np
from nptyping import NDArray

from parllel.arrays import Array
from parllel.buffers import NamedArrayTupleClass
from parllel.samplers import Samples, EnvSamples

from .transform import Transform


@njit
def compute_valid_from_done(done: NDArray[np.bool_], out_valid: NDArray[np.bool_]):
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


class ComputeValidLearningSteps(Transform):
    def __init__(self) -> None:
        """Adds a field to samples buffer under `env.valid` which defines
        whether the time step is valid for learning in recurrent problems.
        Because Pytorch recurrent models cannot reset their hidden state in
        the middle of a batch, steps after an environment reset are invalid
        and must be masked out of the loss.
        """
        self._EnvSamplesClass = None
        
    def dry_run(self, batch_samples: Samples, ArrayCls: Array) -> Samples:
        # get convenient local references
        env_samples: EnvSamples = batch_samples.env
        done = env_samples.done

        EnvSamplesClass = NamedArrayTupleClass(
            typename = env_samples._typename,
            fields = env_samples._fields + ("valid",)
        )

        # allocate new Array objects for advantage and return_
        valid = ArrayCls(shape=done.shape, dtype=done.dtype)

        env_samples = EnvSamplesClass(
            **env_samples._asdict(), valid=valid,
        )

        batch_samples = batch_samples._replace(env = env_samples)
        return self.__call__(batch_samples)

    def __call__(self, batch_samples: Samples) -> Samples:
        compute_valid_from_done(
            np.asarray(batch_samples.env.done),
            np.asarray(batch_samples.env.valid),
        )
        return batch_samples
