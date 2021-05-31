from __future__ import annotations
import ctypes
import multiprocessing as mp
from typing import Dict

import numpy as np

from .array import Array
from .rotating import RotatingArray

class SharedMemoryArray(Array):
    def initialize(self) -> None:
        # allocate array in OS shared memory
        size = int(np.prod(self.shape))
        nbytes = size * np.dtype(self.dtype).itemsize
        # mp.RawArray can be safely passed between processes on startup, even
        # when using the "spawn" start method. However, it cannot be sent
        # through a Pipe or Queue
        self._raw_array = mp.RawArray(ctypes.c_char, nbytes)

        self._wrap_raw_array()
        
    def _wrap_raw_array(self) -> None:
        size = int(np.prod(self.shape))
        self._array = np.frombuffer(self._raw_array, dtype=self.dtype, count=size)

        # assign to shape attribute so that error is raised when data is copied
        # array.reshape might silently copy the data
        self._array.shape = self.shape

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        # remove this numpy array which cannot be pickled
        del state["_array"]
        return state

    def __setstate__(self, state: Dict) -> None:
        # restore state dict entries
        self.__dict__.update(state)
        # restore numpy array
        self._wrap_raw_array()


class RotatingSharedMemoryArray(RotatingArray, SharedMemoryArray):
    pass