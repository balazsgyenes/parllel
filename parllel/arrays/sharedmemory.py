import ctypes
import multiprocessing as mp
from typing import Dict, Tuple

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
        # mp.RawArray can be safely passed between processes on startup, even
        # when using the "spawn" start method. However, it cannot be sent
        # through a Pipe or Queue
        self._raw_array = mp.RawArray(ctypes.c_char, nbytes)

        self._wrap_raw_array()
        
    def _wrap_raw_array(self) -> None:
        size = int(np.prod(self._shape))
        array = np.frombuffer(self._raw_array, dtype=self._dtype, count=size)

        # assign to shape attribute so that error is raised when data is copied
        # array.reshape might silently copy the data
        array.shape = self._shape

        # cast to ViewAwareArray so that indexing operations can be recorded
        self._array: ViewAwareArray = array.view(ViewAwareArray)

    @property
    def unique_id(self) -> int:
        return self._unique_id

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        # remove this numpy array which cannot be pickled
        del state["_array"]
        # add view_locations stored in _array
        state["view_locations"] = self._array.view_locations
        return state

    def __setstate__(self, state: Dict) -> None:
        # remove view_locations from state dictionary
        view_locations = state.pop("view_locations")
        # restore all other state dict entries
        self.__dict__.update(state)
        # restore numpy array
        self._wrap_raw_array()
        self._array.view_locations = view_locations


class RotatingSharedMemoryArray(RotatingArray, SharedMemoryArray):
    pass