from __future__ import annotations
from typing import Any, Tuple

import numpy as np
from nptyping import NDArray

from .array import Array
from parllel.buffers.buffer import Index, Indices

class RotatingArray(Array):
    """An array with padding at both edges of the leading dimension. Calling
    `rotate` copies the data from the padding at the end to the padding at the
    beginning.
    
    Use this array for values like observations, where the last value of the
    previous batch becomes the first value of the next batch. See example:
        >>> T, B = 4, 5
        >>> arr = RotatingArray(shape=(T, B), dtype=int, padding=1)
        >>> arr[T] = np.ones((B,))  # write to last row
        >>> arr.rotate()  # bring the last row to the front of array
        >>> print(arr[0])
        array([1, 1, 1, 1, 1])

    Todo:
        - Add a nice __repr__ function to print head, body, and tail of the array. 
    """
    def __init__(self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        padding: int = 1,
    ) -> None:

        assert padding > 0, "Padding must be positive."
        self._padding = padding
        self._apparent_shape = shape

        # add padding onto both ends of first dimension
        shape = (shape[0] + 2 * self._padding,) + shape[1:]

        super().__init__(shape, dtype)

    def rotate(self) -> None:
        """Prepare buffer for collecting next batch. Rotate values stored at
        the end of buffer for the next batch to become previous values for
        upcoming batch (e.g. value_T becomes value_(-1) and value_(T+1) becomes
        value_0).
        """
        if self._padding > 0:
            final_values = slice(-(self._padding + 1), None)
            next_previous_values = slice(0, self._padding + 1)
            self._array[next_previous_values] = self._array[final_values]

    def __getitem__(self, location: Indices) -> RotatingArray:
        """Index into contained array, applying offset due to padding. The
        result of this method will always have padding of 0, since the padding
        is stripped away.
        """
        if isinstance(location, tuple):
            leading, trailing = location[0], location[1:]
        else:
            leading, trailing = location, ()

        leading = shift_index(leading, self._padding, self._apparent_shape[0])
        result: RotatingArray = super().__getitem__(leading + trailing)
        # modify additional instance variables from RotatingArray
        result._padding = 0
        result._apparent_shape = result.shape
        return result

    def __setitem__(self, location: Indices, value: Any) -> None:
        if not self._padding:
            return super().__setitem__(location, value)
        
        if isinstance(location, tuple):
            leading, trailing = location[0], location[1:]
        else:
            leading, trailing = location, ()

        leading = shift_index(leading, self._padding, self._apparent_shape[0])
        super().__setitem__(leading + trailing, value)

    def __array__(self, dtype = None) -> NDArray:
        array = super().__array__(dtype)
        if self._padding > 0:
            array = array[self._padding:-self._padding]
        return array

    @property
    def start(self) -> int:
        return -self._padding

    @property
    def end(self) -> int:
        return self.shape[0] - self._padding - 1


def shift_index(index: Index, shift: int, apparent_length: int) -> Tuple[Index, ...]:
    """Shifts an array index up by an integer value.
    """
    if isinstance(index, int):
        return (index + shift,)
    if isinstance(index, slice):
        start, stop, step = index.indices(apparent_length)
        return (slice(
            start + shift,
            stop + shift,
            step,
        ),)
    if index is Ellipsis:
        return (slice(shift, -shift), Ellipsis)
    raise ValueError(index)
