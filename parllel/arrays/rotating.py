from typing import Union, Tuple, Any

import numpy as np

from .array import Array

class RotatingArray(Array):
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
        padding: int = 0,
    ) -> None:
        self._shape = shape
        assert dtype is not np.object_, "Data type should not be object."
        self._dtype = dtype
        assert padding > 0, "Padding must be positive."
        self._padding = padding

    def initialize(self):
        # initialize array
        self._shape = (self._shape[0] + 2 * self._padding,) + self._shape[1:]
        super().initialize()

    def rotate(self):
        """Prepare buffer for collecting next batch. Rotate values stored at
        the end of buffer for the next batch to become previous values for
        upcoming batch (e.g. value_T becomes value_(-1) and value_(T+1) becomes
        value_0).
        """
        final_values = slice(-(self._padding + 1), None)
        next_previous_values = slice(0, self._padding + 1)
        self._array[next_previous_values] = self._array[final_values]

    def __getitem__(self, location: Union[Union[int, slice, Ellipsis], Tuple[Union[int, slice, Ellipsis]]]):
        if isinstance(location, Tuple):
            leading, trailing = location[0], location[1:]
        else:
            leading, trailing = location, ()

        leading = shift_index(leading, self._padding)
        return super().__getitem__((leading,) + trailing)

    def __setitem__(self, location: Union[Union[int, slice, Ellipsis], Tuple[Union[int, slice, Ellipsis]]], value: Any):
        if isinstance(location, Tuple):
            leading, trailing = location[0], location[1:]
        else:
            leading, trailing = location, ()

        leading = shift_index(leading, self._padding)
        super().__setitem__((leading,) + trailing, value)

    def __array__(self, dtype = None):
        array = self._array[self._padding:-self._padding]
        if dtype is None:
            return array
        else:
            return array.astype(dtype, copy=False)

    @property
    def start(self):
        return -self._padding

    @property
    def end(self):
        return self._shape[0] - self._padding - 1

def shift_index(index: Union[int, slice, Ellipsis], shift: int):
    if isinstance(index, int):
        index += shift
    elif isinstance(index, slice):
        index = slice(
            start=index.start + shift,
            stop=index + shift,
            step=index.step,
        )
    elif index is Ellipsis:
        index = slice(
            start=shift,
            stop=-shift,
        )
    return index