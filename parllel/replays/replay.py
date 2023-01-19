from typing import Generic, Iterator, Optional, TypeVar

from numpy import random

from parllel.buffers import NamedArrayTuple
from parllel.types import BatchSpec


BufferType = TypeVar("BufferType")


class ReplayBuffer(Generic[BufferType]):
    def __init__(self,
        buffer: NamedArrayTuple,
        sampler_batch_spec: BatchSpec,
        size_T: int, # TODO: infer from inputs
        replay_batch_size: int,
        newest_n_samples_invalid: int = 0,
        oldest_n_samples_invalid: int = 0,
    ) -> None:
        """Stores more than a batch's worth of samples in a circular buffer for
        off-policy algorithms to sample from.
        """
        # convert to ndarray because Array cannot handle indexing with an array
        self.buffer = buffer
        self.batch_spec = sampler_batch_spec
        self.size_T = size_T
        self.batch_size = replay_batch_size
        self.newest_n_samples_invalid = newest_n_samples_invalid
        self.oldest_n_samples_invalid = oldest_n_samples_invalid

        self._cursor: int = 0 # index of next sample to write
        self._full = False # has the entire buffer been written to at least once?
        
        self.seed()
    
    def seed(self, seed: Optional[int] = None) -> None:
        # TODO: replace with seeding module
        self._rng = random.default_rng(seed)

    @property
    def replay_batch_size(self) -> int:
        return self.batch_size

    @property
    def capacity(self):
        return self.size_T * self.batch_spec.B

    def sample_batch(self) -> BufferType:
        if self._full:
            # valid region of buffer wraps around
            # sample integers from 0 to L, and then offset them while wrapping around
            offset = self._cursor + self.oldest_n_samples_invalid
            valid_length = (
                self.size_T
                - self.oldest_n_samples_invalid
                - self.newest_n_samples_invalid
            )
            T_idxs = self._rng.integers(0, valid_length, size=(self.batch_size,))
            T_idxs = (T_idxs + offset) % self.size_T
        else:
            valid_length = self._cursor - self.newest_n_samples_invalid
            T_idxs = self._rng.integers(0, valid_length, size=(self.batch_size,))

        B_idxs = self._rng.integers(0, self.batch_spec.B, size=(self.batch_size,))

        return self.buffer[T_idxs, B_idxs]

    def batches(self) -> Iterator[BufferType]:
        while True:
            yield self.sample_batch()

    def __iter__(self) -> Iterator[BufferType]:
        yield from self.batches
    
    def next_iteration(self) -> None:
        # move cursor forward
        self._cursor += self.batch_spec.T

        if self._cursor >= self.size_T:
            self._full = True
            self._cursor %= self.size_T
