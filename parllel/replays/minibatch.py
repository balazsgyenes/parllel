from typing import Optional

import numpy as np

from parllel.buffers import Buffer, buffer_asarray
from parllel.types import BatchSpec


class MinibatchBuffer:
    def __init__(self,
        buffer: Buffer,
        batch_spec: BatchSpec,
        n_minibatches: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.buffer = buffer
        self.batch_spec = batch_spec
        self.n_minibatches = n_minibatches
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data = buffer_asarray(self.buffer)
        self.size = self.batch_spec.size
        self.minibatch_size = self.size // self.n_minibatches
        if self.drop_last:
            self.upper_limit = self.size - self.minibatch_size + 1
        else:
            self.upper_limit = self.size

    def seed(self, seed: Optional[int] = None):
        # TODO: replace with seeding module
        self._rng = np.random.default_rng(seed)

    def iter_minibatches(self):
        if self.shuffle:
            all_indices = np.arange(self.size)
            self._rng.shuffle(all_indices)

        for start in range(0, self.upper_limit, self.minibatch_size):
            minibatch_indices = slice(start, start + self.minibatch_size)
            if self.shuffle:
                minibatch_indices = all_indices[minibatch_indices]
            
            B_idxs = minibatch_indices // self.batch_spec.T
            T_idxs = minibatch_indices % self.batch_spec.T
            
            minibatch = self.data[T_idxs, B_idxs]
            yield minibatch
