from typing import Any, Tuple, Union

import numpy as np
from nptyping import NDArray

from .array import Array, Index, Indices


class RotatingArray(Array):
    """Abstracts memory management for large arrays.

    Optional padding can be used to extend the first dimension of the array symmetrically on both ends.
    When reusing the buffer, this allows *looking into the past* by indexing out of the regular bounds of the array's shape. 

    Examples:
        >>> a = RotatingArray(shape=(4, 5), dtype=np.int, padding=1)
        >>> a[4, :] = np.ones((1, 5))  # access the last element
        >>> a.rotate()  # bring the last element to the front of the array at position -1
        >>> a[-1]  # In contrast to lists, -1 does not refer to the last element in the array, but to the actual -1st element in the array. 
        array([[1., 1., 1., 1., 1.]])

    Todo:
        - Add a nice __repr__ function to print head, body, and tail of the array. 
    """
    def __init__(self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        padding: int = 1,
    ) -> None:
        super().__init__(shape, dtype)

        assert padding > 0, "Padding must be positive."
        self._padding = padding
        self._apparent_shape = shape

        # add padding onto both ends of first dimension
        self._shape = (self._shape[0] + 2 * self._padding,) + self._shape[1:]

    def rotate(self) -> None:
        """Prepare buffer for collecting next batch. Rotate values stored at
        the end of buffer for the next batch to become previous values for
        upcoming batch (e.g. value_T becomes value_(-1) and value_(T+1) becomes
        value_0).
        """
        final_values = slice(-(self._padding + 1), None)
        next_previous_values = slice(0, self._padding + 1)
        self._array[next_previous_values] = self._array[final_values]

    def __getitem__(self, location: Indices) -> NDArray:
        if isinstance(location, Tuple):
            leading, trailing = location[0], location[1:]
        else:
            leading, trailing = location, ()

        leading = shift_index(leading, self._padding)
        return super().__getitem__((leading,) + trailing)

    def __setitem__(self, location: Indices, value: Any) -> None:
        if isinstance(location, Tuple):
            leading, trailing = location[0], location[1:]
        else:
            leading, trailing = location, ()

        leading = shift_index(leading, self._padding)
        super().__setitem__((leading,) + trailing, value)

    def __array__(self, dtype = None) -> NDArray:
        array = self._array[self._padding:-self._padding]
        if dtype is None:
            return array
        else:
            return array.astype(dtype, copy=False)

    @property
    def start(self) -> int:
        return -self._padding

    @property
    def end(self) -> int:
        return self._shape[0] - self._padding - 1

def shift_index(index: Index, shift: int) -> Union[int, slice]:
    """Shifts an array index up by an integer value.
    """
    if isinstance(index, int):
        index += shift
    elif isinstance(index, slice):
        index = slice(
            index.start + shift,
            index.stop + shift,
            index.step,
        )
    elif index is Ellipsis:
        index = slice(
            shift,
            -shift,
        )
    else:
        raise ValueError(index)
    return index
