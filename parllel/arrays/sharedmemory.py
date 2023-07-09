import ctypes
import multiprocessing as mp
from typing import Dict

import numpy as np

from .array import Array
from .jagged import JaggedArray


class SharedMemoryArray(Array, storage="shared"):
    """An array in OS shared memory that can be shared between processes on
    process startup only (i.e. process inheritance). Starting processes with
    the `spawn` method is also supported.
    """

    storage = "shared"

    def _allocate(self) -> None:
        # allocate array in OS shared memory
        size = int(np.prod(self._allocate_shape))
        nbytes = size * np.dtype(self.dtype).itemsize
        # mp.RawArray can be safely passed between processes on startup, even
        # when using the "spawn" start method. However, it cannot be sent
        # through a Pipe or Queue
        self._raw_array = mp.RawArray(ctypes.c_char, nbytes)

        self._wrap_raw_array()

    def _wrap_raw_array(self) -> None:
        size = int(np.prod(self._allocate_shape))
        self._base_array: np.ndarray = np.frombuffer(
            self._raw_array,
            dtype=self.dtype,
            count=size,
        )

        # assign to shape attribute so that error is raised when data is
        # copied array.reshape might silently copy the data
        self._base_array.shape = self._allocate_shape

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        # remove this numpy array which cannot be pickled
        del state["_base_array"]
        # subprocesses should not be able to call rotate()
        # if processes are started by fork, this is not guaranteed to be called
        state["_rotatable"] = False
        return state

    def __setstate__(self, state: Dict) -> None:
        # restore state dict entries
        self.__dict__.update(state)
        # restore _base_array array
        self._wrap_raw_array()


class SharedMemoryJaggedArray(
    SharedMemoryArray,
    JaggedArray,
    kind="jagged",
    storage="shared",
):
    def _allocate(self) -> None:
        # allocate array in OS shared memory
        size = int(np.prod(self._allocate_shape))
        nbytes = size * np.dtype(self.dtype).itemsize
        # mp.RawArray can be safely passed between processes on startup, even
        # when using the "spawn" start method. However, it cannot be sent
        # through a Pipe or Queue
        self._raw_array = mp.RawArray(ctypes.c_char, nbytes)

        base_batch_shape = self._base_shape[: self._n_batch_dim]
        ptr_shape = base_batch_shape[1:] + (base_batch_shape[0] + 1,)
        ptr_size = int(np.prod(ptr_shape))
        ptr_bytes = ptr_size * np.dtype(np.int64).itemsize
        self._raw_ptr = mp.RawArray(ctypes.c_char, ptr_bytes)

        self._wrap_raw_array()

    def _wrap_raw_array(self) -> None:
        size = int(np.prod(self._allocate_shape))
        self._base_array: np.ndarray = np.frombuffer(
            self._raw_array,
            dtype=self.dtype,
            count=size,
        )

        # assign to shape attribute so that error is raised when data is
        # copied array.reshape might silently copy the data
        self._base_array.shape = self._allocate_shape

        base_batch_shape = self._base_shape[: self._n_batch_dim]
        ptr_shape = base_batch_shape[1:] + (base_batch_shape[0] + 1,)
        ptr_size = int(np.prod(ptr_shape))
        self._ptr: np.ndarray = np.frombuffer(
            self._raw_ptr,
            dtype=np.int64,
            count=ptr_size,
        )
        self._ptr.shape = ptr_shape

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        # remove these numpy arrays which cannot be pickled
        del state["_base_array"]
        del state["_ptr"]
        # arrays should not be rotatable from subprocesses
        state["_rotatable"] = False
        return state

    def __setstate__(self, state: Dict) -> None:
        # restore state dict entries
        self.__dict__.update(state)
        # restore _base_array array
        self._wrap_raw_array()
