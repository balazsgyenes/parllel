from typing import Optional

import numpy as np
from numpy import random

from parllel.arrays import Array
from parllel.buffers import Buffer, Samples, buffer_asarray
from parllel.types import BatchSpec


class ReplayBuffer:
    def __init__(self,
        buffer_to_append: Samples,
        buffer_to_sample: Buffer,
        batch_spec: BatchSpec,
        length_T: int,
        newest_n_samples_invalid: int = 0,
        oldest_n_samples_invalid: int = 0,
    ) -> None:
        """Stores more than a batch's worth of samples in a circular buffer for
        off-policy algorithms to sample from.
        """
        # convert to ndarray because Array cannot handle indexing with an array
        self.append_buffer = buffer_asarray(buffer_to_append)
        self.sample_buffer = buffer_asarray(buffer_to_sample)
        self.batch_spec = batch_spec
        self.length = length_T
        self.newest_n_samples_invalid = newest_n_samples_invalid
        self.oldest_n_samples_invalid = oldest_n_samples_invalid

        self.size = self.length * self.batch_spec.B

        self._cursor: int = 0 # index of next sample to write
        self._full = False # has the entire buffer been written to at least once?
        
        self.seed()
    
    def seed(self, seed: Optional[int] = None) -> None:
        # TODO: replace with seeding module
        self._rng = random.default_rng(seed)

    def sample_batch(self, n_samples: int) -> Buffer[np.ndarray]:
        if self._full:
            # valid region of buffer wraps around
            # sample integers from 0 to L, and then offset them while wrapping around
            offset = self._cursor + self.oldest_n_samples_invalid
            L = (
                self.length
                - self.oldest_n_samples_invalid
                - self.newest_n_samples_invalid
            )
            T_idxs = self._rng.integers(0, L, size=(n_samples,))
            T_idxs = (T_idxs + offset) % self.length
        else:
            T_idxs = self._rng.integers(0, self._cursor, size=(n_samples,))

        B_idxs = self._rng.integers(0, self.batch_spec.B, size=(n_samples,))

        return self.sample_buffer[T_idxs, B_idxs]

    def append_samples(self, samples: Samples[Array]) -> None:

        if self._cursor + self.batch_spec.T > self.length:
            # indices where samples are inserted wrap around end of buffer
            idxs = np.arange(self._cursor, self._cursor + self.batch_spec.T) % self.length
        else:
            idxs = slice(self._cursor, self._cursor + self.batch_spec.T)
        
        # TODO: add ability for replay buffer and batch buffer to be different
        # TODO: without explicitly writing to the padding of the observation
        # buffer, there is no way to set the next_observation for the last step
        # in the replay buffer
        self.append_buffer[idxs] = samples

        # move cursor forward
        self._cursor += self.batch_spec.T

        if self._cursor >= self.length:
            # note that previous check is for greater than, but here we also
            # check for equality
            self._full = True
            self._cursor %= self.length
        