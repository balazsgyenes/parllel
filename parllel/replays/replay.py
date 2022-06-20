from typing import Any, Optional

import numpy as np
from numpy import random

from parllel.buffers import Buffer, Indices
from parllel.types import BatchSpec

class ReplayBuffer(Buffer):
    def __init__(self, buffer: Buffer, batch_spec: BatchSpec) -> None:
        self._buffer = buffer
        self._batch_spec = batch_spec
        
        self._size = #TODO

        # only samples between _begin:_end are valid
        self._begin: int = 0
        self._end: int = 0
        self._full = False # has the entire buffer been written to at least once?
        
        self.seed() # TODO: replace with seeding module
    
    def seed(self, seed: Optional[int] = None):
        self._rng = random.default_rng(seed)

    def __getattr__(self, name: str) -> Any:
        if "_buffer" in self.__dict__:
            return getattr(self._buffer, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setitem__(self, location: Indices, value: Any):
        # TODO: is this needed?
        pass

    def sample_batch(self):
        pass

    def rotate(self):
        self._end = (self._end + self._batch_spec.T) % self._size
        
        if self._full:
            self._begin = (self._begin + self._batch_spec.T) % self._size

        if self._end == 0:
            # on next rotate, begin needs to be incremented too
            self._full = True
