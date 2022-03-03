from __future__ import annotations
import ctypes
from functools import reduce
import multiprocessing as mp
from typing import Dict

import numpy as np

from .array import Array
from .rotating import RotatingArray

class SharedMemoryArray(Array):
    """An array in OS shared memory that can be shared between processes on
    process startup only (i.e. process inheritance). Starting processes with
    the `spawn` method is also supported.
    """
    def _allocate(self) -> None:
        # allocate array in OS shared memory
        size = int(np.prod(self._base_shape))
        nbytes = size * np.dtype(self.dtype).itemsize
        # mp.RawArray can be safely passed between processes on startup, even
        # when using the "spawn" start method. However, it cannot be sent
        # through a Pipe or Queue
        self._raw_array = mp.RawArray(ctypes.c_char, nbytes)

        self._wrap_raw_array()
        
    def _wrap_raw_array(self) -> None:
        size = int(np.prod(self._base_shape))
        self._array = np.frombuffer(self._raw_array, dtype=self.dtype, count=size)

        # assign to shape attribute so that error is raised when data is copied
        # array.reshape might silently copy the data
        self._array.shape = self._base_shape

        if self._index_history:
            if len(self._index_history) > 1:
                # TODO: not correct for rotating arrays
                raise NotImplementedError
            self._array = reduce(lambda buf, index: buf[index],
                                 self._index_history[:-1], self._array)

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