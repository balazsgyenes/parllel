from functools import reduce
from operator import getitem
from typing import Any, Tuple

import numpy as np

from parllel.buffers import Index, Indices

from .array import Array, compute_current_indices


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
        - Add a nice __repr__ function to print head, body, and tail of the
            array.
    """
    def __init__(self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        padding: int = 1,
    ) -> None:

        if not padding > 0:
            raise ValueError("Padding must be positive.")
        self._padding = padding

        if not shape:
            raise ValueError("Non-empty shape required.")

        if not shape[0] >= padding:
            raise ValueError(f"Leading dimension {shape[0]} must be at least "
                f"as long as padding {padding}")
        
        # add padding onto both ends of first dimension
        padded_shape = (shape[0] + 2 * self._padding,) + shape[1:]

        super().__init__(shape=padded_shape, dtype=dtype)

        # unlike in base class, _current_array is not the same as _base_array
        # this also fixes _apparent_shape, which still includes the padding
        self._resolve_indexing_history()

    @property
    def padding(self) -> int:
        return self._padding

    @property
    def first(self) -> int:
        """The index of the first element in the array, not including padding.
        Enables syntactic sugar like `arr[arr.first - 1]`
        """
        return 0

    @property
    def last(self) -> int:
        """The index of the final element in the array, not including padding.
        Replaces indexing at -1 in numpy arrays.
        e.g. array[-1] -> rot_array[rot_array.last]
        """
        return self._apparent_shape[0] - 1

    def _resolve_indexing_history(self) -> None:
        if self._index_history:
            # shift only the first indices, leave the rest (if thereare more)
            index_history = [shift_indices(self._index_history[0], self._padding),
                            ] + self._index_history[1:]
        else:
            # even if the array was never indexed, only this slice of the array
            # should be returned by __array__
            index_history = [slice(self._padding, -self._padding)]

        # if index history has only 1 element, this has no effect
        self._current_array = reduce(getitem, index_history, self._base_array)
        self._apparent_shape = self._current_array.shape
        self._current_indices = compute_current_indices(
            self._base_array, self._current_array)

    def __setitem__(self, location: Indices, value: Any) -> None:
        if self._index_history:
            return super().__setitem__(location, value)
        
        location = shift_indices(location, self._padding)
        self._base_array[location] = value

    def rotate(self) -> None:
        """Rotate values stored at the end of buffer for the next batch to
        become previous values for upcoming batch. Usually called to prepare
        buffer for collecting next batch.

        before rotate()     ->  after rotate()
        -------------------------------------------------------
        value[last + 2]     ->  value[first + 1]    = value[1]
        value[last + 1]     ->  value[first]        = value[0]
        value[last]         ->  value[first - 1]    = value[-1]
        value[last - 1]     ->  value[first - 2]    = value[-2]
        """
        if not self._index_history:
            # only rotate if called on the base array.
            # rotating subarrays is not possible anyway
            final_values = slice(-(self._padding * 2), None)
            next_previous_values = slice(0, self._padding * 2)
            self._base_array[next_previous_values] = self._base_array[final_values]


def shift_indices(indices: Indices, shift: int) -> Tuple[Index, ...]:
    if isinstance(indices, tuple):
        first, rest = indices[0], indices[1:]
    else:
        first, rest = indices, ()
    return shift_index(first, shift) + rest


def shift_index(index: Index, shift: int) -> Tuple[Index, ...]:
    """Shifts an array index up by an integer value.
    """
    if isinstance(index, int):
        if index < -shift:
            raise IndexError(f"Not enough padding ({shift}) to accomodate "
                             f"index ({index})")
        return (index + shift,)
    if isinstance(index, slice):
        # in case the step is negative, we need to reverse/adjust the limits
        # limits must be incremented because the upper limit of the slice is
        # not in the slice
        # [:] = slice(None, None, None) -> slice(shift, -shift, None)
        # [::-1] = slice(None, None, -1) -> slice(-shift-1, shift-1, -1)
        # [:3:-1] = slice(None, 3, -1) -> slice(-shift-1, 3+shift, -1)
        lower_limit = -(shift+1) if index.step is not None and index.step < 0 else shift
        upper_limit = shift-1 if index.step is not None and index.step < 0 else -shift
        return (slice(
            index.start + shift if index.start is not None else lower_limit,
            index.stop + shift if index.stop is not None else upper_limit,
            index.step,
        ),)
    if index is Ellipsis:
        # add another Ellipsis, to index any remaining dimensions that an
        # Ellipsis would have indexed (possible no extra dimensions remain)
        return (slice(shift, -shift), Ellipsis)
    raise ValueError(index)
