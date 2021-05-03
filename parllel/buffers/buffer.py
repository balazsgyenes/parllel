import ctypes
import multiprocessing as mp
from typing import Union, Tuple

import numpy as np


class Buffer:
    """Abstracts memory management for large arrays.
    """
    def __init__(self,
        shape: Tuple[int],
        dtype: np.dtype,
        shared_memory: bool = False,
        padding: int = 0
    ):
        self._shape = shape
        assert dtype is not np.object_, "Data type should not be object."
        self._dtype = dtype
        self._shared_memory = shared_memory
        assert padding >= 0, "Padding must not be negative."
        self._padding = padding

    def initialize(self):
        # initialize buffer in either local or shared memory
        padded_shape = (self._shape[0] + 2 * self._padding,) + self._shape[1:]
        if self._shared_memory:
            size = int(np.prod(padded_shape))
            nbytes = size * np.dtype(self._dtype).itemsize
            mp_array = mp.RawArray(ctypes.c_char, nbytes)
            self._buffer = np.frombuffer(mp_array, dtype=self._dtype, count=size)
            # assign to shape attribute so that error is raised when data is copied
            # _buffer.reshape might silently copy the data
            self._buffer.shape = padded_shape
        else:
            self._buffer = np.zeros(shape=padded_shape, dtype=self._dtype)
        
        # store position of last write operation. This is used by `rotate()` to
        # ensure that the last write from the last batch ends up as previous
        # value.
        self._position = 0

    def rotate(self):
        """Prepare buffer for collecting next batch.
        """
        if self._padding > 0:
            """Rotate values stored at the end of buffer for the next batch to
            become previous values for upcoming batch (e.g. value_T becomes
            value_(-1) and value_(T+1) becomes value_0).
            """
            last_values = slice(self._position, self._position + self._padding + 1)
            next_previous_values = slice(0, self._padding + 1)
            self._buffer[next_previous_values] = self._buffer[last_values]
        self._position = 0

    def __getitem__(self, location: Union[int, slice]):
        if isinstance(location, int):
            return self._buffer[location + self._padding]
        else:
            raise NotImplementedError

    def __setitem__(self, location: Union[int, slice], value):
        self._position += 1
        if isinstance(location, int):
            self._buffer[location + self._padding] = value
        else:
            raise NotImplementedError
    
    def __array__(self, dtype = None):
        """Called to convert to numpy array.

        Docs: https://numpy.org/devdocs/user/basics.dispatch.html
        """
        raise NotImplementedError

    @property
    def start(self):
        return -self._padding

    @property
    def end(self):
        return self._shape[0] + 2 * self._padding