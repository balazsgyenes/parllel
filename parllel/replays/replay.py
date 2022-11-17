from typing import Generic, Iterator, Optional, TypeVar

import numpy as np
from numpy import random

from parllel.arrays import Array
from parllel.buffers import Buffer, Samples, buffer_asarray, NamedArrayTuple
from parllel.types import BatchSpec


BufferType = TypeVar("BufferType")


class ReplayBuffer(Generic[BufferType]):
    def __init__(self,
        buffer: NamedArrayTuple,
        sampler_batch_spec: BatchSpec,
        leading_dim: int,
        n_samples: int,
        newest_n_samples_invalid: int = 0,
        oldest_n_samples_invalid: int = 0,
    ) -> None:
        """Stores more than a batch's worth of samples in a circular buffer for
        off-policy algorithms to sample from.
        """
        # convert to ndarray because Array cannot handle indexing with an array
        self.buffer = buffer
        self.batch_spec = sampler_batch_spec
        self.leading_dim = leading_dim
        self._n_samples = n_samples
        self.newest_n_samples_invalid = newest_n_samples_invalid
        self.oldest_n_samples_invalid = oldest_n_samples_invalid

        self.size = self.leading_dim * self.batch_spec.B

        self._cursor: int = 0 # index of next sample to write
        self._full = False # has the entire buffer been written to at least once?
        
        self.seed()
    
    def seed(self, seed: Optional[int] = None) -> None:
        # TODO: replace with seeding module
        self._rng = random.default_rng(seed)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def __len__(self) -> int:
        return self.size

    def sample_batch(self) -> BufferType:
        if self._full:
            # valid region of buffer wraps around
            # sample integers from 0 to L, and then offset them while wrapping around
            offset = self._cursor + self.oldest_n_samples_invalid
            L = (
                self.leading_dim
                - self.oldest_n_samples_invalid
                - self.newest_n_samples_invalid
            )
            T_idxs = self._rng.integers(0, L, size=(self._n_samples,))
            T_idxs = (T_idxs + offset) % self.leading_dim
        else:
            T_idxs = self._rng.integers(0, self._cursor, size=(self._n_samples,))

        B_idxs = self._rng.integers(0, self.batch_spec.B, size=(self._n_samples,))

        return self.buffer[T_idxs, B_idxs]

    def batches(self) -> Iterator[BufferType]:
        while True:
            yield self.sample_batch()

    def __iter__(self) -> Iterator[BufferType]:
        yield from self.batches
    
    def rotate(self) -> None:

        # move cursor forward
        self._cursor += self.batch_spec.T

        if self._cursor >= self.leading_dim:
            # note that previous check is for greater than, but here we also
            # check for equality
            self._full = True
            self._cursor %= self.leading_dim
