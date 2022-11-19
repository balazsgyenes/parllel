from functools import reduce
from typing import Any, Optional, Tuple

import numpy as np

from parllel.buffers import Index, Indices

from .array import Array
from .rotating import RotatingArray


class LargeArray(RotatingArray):

    def __init__(self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        padding: int = 0,
        apparent_size: Optional[int] = None,
    ) -> None:

        if apparent_size is None:
            apparent_size = shape[0]
        if shape[0] % apparent_size != 0:
            raise ValueError(
                "The leading dimension of the array must divide evenly into "
                "the apparent leading dimension."
            )
        self.apparent_size = apparent_size
        # self.large = (shape[0] > apparent_size)
        self.offset = 0

        super().__init__(shape, dtype, padding)

    def _resolve_indexing_history(self) -> None:
        array = self._base_array
        shift = self.offset + self._padding

        if self._index_history:
            # shift only the first indices, leave the rest (if thereare more)
            # TODO: modify this method so it also takes the apparent shape
            # right now, it won't account for the hidden region
            index_history = [shift_indices(
                self._index_history[0],
                shift,
                self.apparent_size,
            )] + self._index_history[1:]
        else:
            # even if the array was never indexed, only this slice of the array
            # should be returned by __array__
            index_history = [slice(shift, shift + self.apparent_size)]

        # if index history has only 1 element, this has no effect
        array = reduce(lambda arr, index: arr[index], index_history[:-1], array)
        self._previous_array = array
        
        # we guarantee that index_history has at least 1 element
        array = array[index_history[-1]]

        self._current_array = array
        self._apparent_shape = array.shape
    
    def __setitem__(self, location: Indices, value: Any) -> None:
        if self._current_array is None:
            self._resolve_indexing_history()

        if self._index_history:
            if self._apparent_shape == ():
                # Need to avoid item assignment on a scalar (0-D) array, so we assign
                # into previous array using the last indices used
                if not (location == slice(None) or location == ...):
                    raise IndexError("Cannot take slice of 0-D array.")
                location = self._index_history[-1]
                # indices must be shifted if they were the first indices
                if len(self._index_history) == 1:
                    location = shift_indices(location, self.offset + self._padding, self.apparent_size)
                destination = self._previous_array
            else:
                destination = self._current_array
        else:
            location = shift_indices(location, self.offset + self._padding, self.apparent_size)
            destination = self._base_array
        destination[location] = value
    
    def rotate(self) -> None:

        self.offset = (self.offset + self.apparent_size) % (
            self._base_shape[0] - 2 * self.padding)

        super().rotate()


def shift_indices(indices: Indices, shift: int, apparent_size: int,
) -> Tuple[Index, ...]:
    if isinstance(indices, tuple):
        first, rest = indices[0], indices[1:]
    else:
        first, rest = indices, ()
    return shift_index(first, shift, apparent_size) + rest


def shift_index(index: Index, shift: int, size: int,
) -> Tuple[Index, ...]:
    """Shifts an array index up by an integer value.
    """
    if shift == 0:
        # TODO: does Ellipsis need to be converted?
        return (index,)
    if isinstance(index, int):
        if index < -shift:
            raise IndexError(
                f"Not enough padding ({shift}) to accomodate index ({index})"
            )
        return (index + shift,)
    if isinstance(index, np.ndarray):
        if np.any(index < -shift):
            raise IndexError(
                f"Not enough padding ({shift}) to accomodate index ({index})"
            )
        return (index + shift,)
    if isinstance(index, slice):
        flipped = index.step is not None and index.step < 0
        # in case the step is negative, we need to reverse/adjust the limits
        # limits must be incremented because the upper limit of the slice is
        # not in the slice
        # [:] = slice(None, None, None) -> slice(shift, shift+size, None)
        # [::-1] = slice(None, None, -1) -> slice(shift+size-1, shift-1, -1)
        # [:3:-1] = slice(None, 3, -1) -> slice(shift+size-1, 3+shift, -1)

        if index.start is not None:
            start = index.start + shift
        else:
            start = shift + size - 1 if flipped else shift
        
        if index.stop is not None:
            stop = index.stop + shift
        else:
            stop = shift - 1 if flipped else shift + size

        return (slice(start, stop, index.step),)
    if index is Ellipsis:
        # add another Ellipsis, to index any remaining dimensions that an
        # Ellipsis would have indexed (possible no extra dimensions remain)
        return (slice(shift, shift + size), Ellipsis)
    raise ValueError(index)
