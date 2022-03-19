from __future__ import annotations
from typing import Tuple

import numpy as np

from .array import Array
from parllel.buffers.buffer import Buffer, Index, Indices


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

        # add padding onto both ends of first dimension
        assert shape, "Non-empty shape required."
        padded_shape = (shape[0] + 2 * self._padding,) + shape[1:]

        super().__init__(shape=padded_shape, dtype=dtype)

        # parent class assigns padded_shape to apparent_shape
        self._apparent_shape = shape

        # initialize array as though the base array had been indexed such that
        # now the padding is removed
        self._index_history = [slice(self._padding, -self._padding)]
        self._resolve_indexing_history()
        # this also fixes _apparent_shape, which still includes the padding

    @Buffer.index_history.getter
    def index_history(self) -> Tuple[Tuple[Index, ...], ...]:
        # return without the first indices which remove the padding
        return self._index_history[1:]

    @property
    def end(self) -> int:
        """The index of the final element in the array, not including padding.
        """
        return self._apparent_shape[0] - 1

    def rotate(self) -> None:
        """Prepare buffer for collecting next batch. Rotate values stored at
        the end of buffer for the next batch to become previous values for
        upcoming batch (e.g. value_T becomes value_(-1) and value_(T+1) becomes
        value_0).
        """
        if not self._index_history:
            # only rotate if called on the base array.
            # rotating subarrays is not possible anyway
            final_values = slice(-(self._padding + 1), None)
            next_previous_values = slice(0, self._padding + 1)
            self._base_array[next_previous_values] = self._base_array[final_values]
