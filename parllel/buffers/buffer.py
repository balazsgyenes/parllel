import ctypes
import multiprocessing as mp
from typing import Union, Tuple

import numpy as np


class Buffer:
    """Abstracts memory management for large arrays.

    Optional padding can be used to extend the first dimension of the array symmetrically on both ends.
    When reusing the buffer, this allows *looking into the past* by indexing out of the regular bounds of the array's shape. 

    Examples:
        >>> a = Buffer(shape=(4, 5), dtype=np.int, padding=1)
        >>> a[4, :] = np.ones((1, 5))  # access the last element
        >>> a.rotate()  # bring the last element to the front of the array at position -1
        >>> a[-1]  # In contrast to lists, -1 does not refer to the last element in the array, but to the actual -1st element in the array. 
        array([[1., 1., 1., 1., 1.]])
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
        
    def rotate(self):
        """Prepare buffer for collecting next batch.
        """
        if self._padding > 0:
            """Rotate values stored at the end of buffer for the next batch to
            become previous values for upcoming batch (e.g. value_T becomes
            value_(-1) and value_(T+1) becomes value_0).
            """
            last_values = slice(-(self._padding + 1), None)
            next_previous_values = slice(0, self._padding + 1)
            self._buffer[next_previous_values] = self._buffer[last_values]

    def __getitem__(self, location: Union[int, slice]):
        if isinstance(location, int):
            return self._buffer[location + self._padding]
        elif isinstance(location, slice):
            shifted_location = slice(
                start=location.start + self._padding,
                stop=location + self._padding,
                step=location.step,
            )
            return self._buffer[shifted_location]
        else:
            raise NotImplementedError

    def __setitem__(self, location: Union[int, slice], value):
        if isinstance(location, int):
            self._buffer[location + self._padding] = value
        elif isinstance(location, slice):
            shifted_location = slice(
                start=location.start + self._padding,
                stop=location + self._padding,
                step=location.step,
            )
            self._buffer[shifted_location] = value
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