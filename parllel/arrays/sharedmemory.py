import ctypes
import multiprocessing as mp
from typing import Tuple

import numpy as np

from .array import Array
from .rotating import RotatingArray
from .view_aware import ViewAwareArray


class SharedMemoryArray(Array):
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype):
        super().__init__(shape, dtype)
    
        # create a unique identifier for this buffer
        # since all buffer objects are created in the main process, we can be
        # sure this is unique, even after child processes are started.
        self._unique_id: int = id(self)

    def initialize(self) -> None:
        # allocate array in OS shared memory
        size = int(np.prod(self._shape))
        nbytes = size * np.dtype(self._dtype).itemsize
        mp_array = mp.RawArray(ctypes.c_char, nbytes)
        array = np.frombuffer(mp_array, dtype=self._dtype, count=size)
        # assign to shape attribute so that error is raised when data is copied
        # array.reshape might silently copy the data
        array.shape = self._shape

        # cast to ViewAwareArray so that indexing operations can be recorded
        self._array = array.view(ViewAwareArray)

    @property
    def unique_id(self) -> int:
        return self._unique_id


class RotatingSharedMemoryArray(RotatingArray, SharedMemoryArray):
    pass