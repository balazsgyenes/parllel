from functools import reduce
from typing import Any, Tuple

import numpy as np

from parllel.buffers import Index, Indices

from .array import Array


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
        # self._apparent_shape = shape

        # initialize array as though the base array had been indexed such that
        # now the padding is removed
        # self._index_history = [slice(self._padding, -self._padding)]
        self._resolve_indexing_history()
        # this also fixes _apparent_shape, which still includes the padding

    # @Buffer.index_history.getter
    # def index_history(self) -> Tuple[Tuple[Index, ...], ...]:
    #     # return without the first indices which remove the padding
    #     return self._index_history[1:]

    @property
    def end(self) -> int:
        """The index of the final element in the array, not including padding.
        """
        return self._apparent_shape[0] - 1

    # def _resolve_indexing_history(self) -> None:
    #     # this function is a little different than the parent class because we
    #     # want previous_array to
    #     array = self._base_array

    #     if self._index_history:
    #         first = self._index_history[0]
    #         first = shift_index(first, self._padding)
    #     else:
    #         first = slice(self._padding, -self._padding)

    #     array = array[first]
        
    #     # if index history has less than 3 elements, this has no effect
    #     array = reduce(lambda arr, index: arr[index], self._index_history[1:-1], array)
    #     self._previous_array = array
        
    #     # index last item in history only if there is a history
    #     if len(self._index_history) > 1:
    #         array = array[self._index_history[-1]]

    #     self._current_array = array
    #     self._apparent_shape = array.shape

    def _resolve_indexing_history(self) -> None:

        if len(self._index_history) == 0:
            default_slice = slice(self._padding, -self._padding)
            self._previous_array = self._base_array[default_slice]
            self._current_array = self._previous_array
        elif len(self._index_history) == 1:
            default_slice = slice(self._padding, -self._padding)
            self._previous_array = self._base_array[default_slice]

            index = shift_index(self._index_history[0], self._padding)
            self._current_array = self._base_array[index]
        else: # len(self._index_history) >= 2
            array = self._base_array
            index = shift_index(self._index_history[0], self._padding)
            array = array[index]

            array = reduce(lambda arr, index: arr[index], self._index_history[1:-1], array)

            self._previous_array = array

            self._current_array = array[self._index_history[-1]]
        self._apparent_shape = self._current_array.shape

    def __setitem__(self, location: Indices, value: Any) -> None:
        if self._index_history:
            return super().__setitem__(location, value)
        
        if isinstance(location, tuple):
            first, rest = location[0], location[1:]
        else:
            first, rest = location, ()
        first = shift_index(first, self._padding)
        self._base_array[first + rest] = value

    def rotate(self) -> None:
        """Prepare buffer for collecting next batch. Rotate values stored at
        the end of buffer for the next batch to become previous values for
        upcoming batch (e.g. value_T becomes value_(-1) and value_(T+1) becomes
        value_0).
        """
        if not self._index_history:
            # only rotate if called on the base array.
            # rotating subarrays is not possible anyway
            final_values = slice(-(self._padding * 2), None)
            next_previous_values = slice(0, self._padding * 2)
            self._base_array[next_previous_values] = self._base_array[final_values]


def shift_index(index: Index, shift: int) -> Tuple[Index, ...]:
    """Shifts an array index up by an integer value.
    """
    if isinstance(index, int):
        return (index + shift,)
    if isinstance(index, slice):
        return (slice(
            index.start + shift if index.start is not None else shift,
            index.stop + shift if index.stop is not None else -shift,
            index.step,
        ),)
    if index is Ellipsis:
        return (slice(shift, -shift), Ellipsis)
    raise ValueError(index)
