import ctypes
import multiprocessing as mp
from typing import Union, Tuple

import numpy as np

from parllel.buffers.buffer import Buffer
from parllel.buffers.weak import WeakBuffer


class SharedMemoryBuffer(Buffer):
    def __init__(self, shape: Tuple[int], dtype: np.dtype, shared_memory: bool, padding: int):
        super().__init__(shape, dtype, shared_memory=shared_memory, padding=padding)
    
        # create a unique identifier for this buffer
        # since all buffer objects are created in the main process, we can be
        # sure this is unique, even after child processes are started.
        self._unique_id = id(self)

    def initialize(self):
        # allocate array in OS shared memory
        padded_shape = (self._shape[0] + 2 * self._padding,) + self._shape[1:]
        size = int(np.prod(padded_shape))
        nbytes = size * np.dtype(self._dtype).itemsize
        mp_array = mp.RawArray(ctypes.c_char, nbytes)
        self._buffer = np.frombuffer(mp_array, dtype=self._dtype, count=size)
        # assign to shape attribute so that error is raised when data is copied
        # _buffer.reshape might silently copy the data
        self._buffer.shape = padded_shape

        # create a lock that will be used to ensure that read/write operations
        # wait for dispatched writes to finish
        self._lock = mp.Lock()

    def __getitem__(self, location: Union[int, slice]):
        self._lock.acquire()
        item = super().__getitem__(location)
        self._lock.release()
        return item

    def __setitem__(self, location: Union[int, slice], value):
        if isinstance(value, WeakBuffer):
            self._lock.acquire()
            value.dispatch_write(target_buffer_id=self.unique_id, location=location)
            # lock is released by process that completes the dispatched write
        else:
            super().__setitem__(location, value)

    @property
    def unique_id(self):
        return self._unique_id