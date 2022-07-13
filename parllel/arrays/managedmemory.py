from multiprocessing import shared_memory
from typing import Dict

import numpy as np

from .array import Array
from .rotating import RotatingArray


class ManagedMemoryArray(Array):
    """An array in OS shared memory that can be shared between processes at any
    time.
    """
    def _allocate(self) -> None:
        # allocate array in OS shared memory
        size = int(np.prod(self._base_shape))
        nbytes = size * np.dtype(self.dtype).itemsize
        # SharedMemory is given a unique name that other processes can use to
        # attach to it.
        self._raw_array = shared_memory.SharedMemory(create=True, size=nbytes)

        self._wrap_raw_array()
        
    def _wrap_raw_array(self) -> None:
        size = int(np.prod(self._base_shape))
        self._base_array = np.frombuffer(self._raw_array.buf, dtype=self.dtype, count=size)

        # assign to shape attribute so that error is raised when data is copied
        # array.reshape might silently copy the data
        self._base_array.shape = self._base_shape

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        # remove arrays which cannot be pickled
        del state["_raw_array"]
        del state["_base_array"]
        del state["_current_array"]
        del state["_previous_array"]
        state["_memory_name"] = self._raw_array.name
        return state

    def __setstate__(self, state: Dict) -> None:
        # restore state dict entries
        name = state.pop("_memory_name")
        self.__dict__.update(state)
        # restore _raw_array
        self._raw_array = shared_memory.SharedMemory(name=name)
        # restore _base_array array
        self._wrap_raw_array()
        # other arrays will be resolved when required
        self._previous_array = None
        self._current_array = None

    def close(self):
        # delete these arrays that contain references to the shared memory
        if hasattr(self, "_base_array"):
            del self._base_array
        if hasattr(self, "_current_array"):
            del self._current_array
        if hasattr(self, "_previous_array"):
            del self._previous_array
        # close must be called by each instance (i.e. each process) on cleanup
        self._raw_array.close()

    def destroy(self):
        # unlink must be called once and only once to release shared memory
        # TODO: make sure this called properly
        self._raw_array.unlink()


class RotatingManagedMemoryArray(RotatingArray, ManagedMemoryArray):
    pass
