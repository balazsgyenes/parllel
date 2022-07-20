from typing import Any, Optional

import numpy as np
from numpy import random

from parllel.buffers import Buffer, Samples, NamedArrayTupleClass, buffer_asarray
from parllel.types import BatchSpec


SarsSamples = NamedArrayTupleClass("SarsSample",
    ["observation", "action", "reward", "done", "next_observation"])


class ReplayBuffer:
    def __init__(self,
        buffer: Buffer,
        batch_spec: BatchSpec,
        length_T: int,
    ) -> None:
        """Stores more than a batch's worth of samples in a circular buffer for
        off-policy algorithms to sample from.
        """
        self.buffer = buffer
        self.batch_spec = batch_spec
        
        self.length = length_T
        self.size = self.length * self.batch_spec.B

        # TODO: replace these hard-coded values
        self.invalid_samples_at_front = 1 # next_observation not set yet
        # actually, all samples have a next_observation already, but it is not
        # copied into the replay buffer because of conversion to ndarray
        self.invalid_samples_at_back = 0

        # only samples between _begin:_end are valid
        self._begin: int = 0
        self._end: int = 0
        self._full = False # has the entire buffer been written to at least once?
        
        self.seed()
    
    def seed(self, seed: Optional[int] = None):
        # TODO: replace with seeding module
        self._rng = random.default_rng(seed)

    def sample_batch(self, n_samples):
        begin = self._begin + self.invalid_samples_at_back
        end = self._end - self.invalid_samples_at_front

        if begin > end:
            # valid region of buffer wraps around
            # sample integers from 0 to L, and then offset them while wrapping around
            L = self.length + end - begin
            T_idxs = self._rng.integers(0, L, size=(n_samples,))
            T_idxs = (T_idxs + begin) % self.length
        else:
            T_idxs = self._rng.integers(begin, end, size=(n_samples,))

        B_idxs = self._rng.integers(0, self.batch_spec.B, size=(n_samples,))

        # TODO: move this to user-defined function, currently hard-coded
        observation = self.buffer.env.observation

        samples = SarsSamples(
            observation=observation,
            action=self.buffer.agent.action,
            reward=self.buffer.env.reward,
            done=self.buffer.env.done,
            # TODO: replace with observation.next
            next_observation=observation[1 : observation.last + 2],
        )

        samples = buffer_asarray(samples)
        samples = samples[T_idxs, B_idxs]

        return samples

    def append_samples(self, samples: Samples):

        if self._end + self.batch_spec.T > self.length:  # Wrap.
            idxs = np.arange(self._end, self._end + self.batch_spec.T) % self.length
            # samples at beginning are now being overwritten
            # from now on, begin needs to be incremented too
            self._full = True
        else:
            idxs = slice(self._end, self._end + self.batch_spec.T)
        
        # # TODO: add ability for replay buffer and batch buffer to be different
        self.buffer[idxs] = samples

        # move cursor forward
        self._end = (self._end + self.batch_spec.T) % self.length
        
        if self._full:
            self._begin = (self._begin + self.batch_spec.T) % self.length
