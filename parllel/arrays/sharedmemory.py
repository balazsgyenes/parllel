import ctypes
import multiprocessing as mp
from typing import Any

import numpy as np

from .array import Array


class SharedMemoryArray(Array, storage="shared"):
    """An array in OS shared memory that can be shared between processes on
    process startup only (i.e. process inheritance). Starting processes with
    the `spawn` method is also supported.
    """

    storage = "shared"

    def _allocate(self, shape: tuple[int, ...], dtype: np.dtype, name: str) -> None:
        # allocate array in OS shared memory
        size = int(np.prod(shape))
        nbytes = size * np.dtype(dtype).itemsize
        # mp.RawArray can be safely passed between processes on startup, even
        # when using the "spawn" start method. However, it cannot be sent
        # through a Pipe or Queue
        raw_array = mp.RawArray(ctypes.c_char, nbytes)

        # store allocation for getstate and setstate methods
        if not hasattr(self, "_allocations"):
            self._allocations = []
        self._allocations.append((name, raw_array, shape, dtype))

        # wrap RawArray with a numpy array
        array = self._wrap_with_ndarray(raw_array, shape, dtype)

        # assign to requested attribute name
        setattr(self, name, array)

    @staticmethod
    def _wrap_with_ndarray(
        raw_array,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray:
        size = int(np.prod(shape))
        array: np.ndarray = np.frombuffer(raw_array, dtype=dtype, count=size)
        # assign to shape attribute so that error is raised when data is copied
        # array.reshape might silently copy the data
        array.shape = shape
        return array

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # subprocesses should not be able to call rotate()
        # if processes are started by fork, this is not guaranteed to be called
        state["_rotatable"] = False

        # remove allocated numpy arrays which cannot be pickled
        for allocation in self._allocations:
            name = allocation[0]
            del state[name]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # restore state dict entries
        self.__dict__.update(state)

        # restore numpy arrays by re-wrapping them
        for allocation in self._allocations:
            name, raw_array, shape, dtype = allocation
            array = self._wrap_with_ndarray(raw_array, shape, dtype)
            setattr(self, name, array)
