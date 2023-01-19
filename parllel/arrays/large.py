from __future__ import annotations # full returns another LargeArray
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
        self.full_size = shape[0] # full leading dimension without padding
        self.apparent_size = apparent_size
        self.offset = 0
        self.shift = self.offset + padding

        super().__init__(shape, dtype, padding)
        
    def _resolve_indexing_history(self) -> None:
        array = self._base_array
        self.shift = shift = self.offset + self._padding

        if self._index_history:
            # shift only the first indices, leave the rest (if there are more)
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
    
    def reset(self) -> None:
        """Resets array, such after calling `rotate()` once, the offset will
        be 0. This is useful in the sampler, which calls `rotate()` before
        every batch.
        """
        if self._index_history:
            raise RuntimeError("Only allowed to call `reset()` on original array")
        
        # if apparent size is not smaller, sets offset to 0
        self.offset = self.full_size - self.apparent_size

        # current array is now invalid, but apparent shape should still be
        # correct
        self._current_array = None

    def __setitem__(self, location: Indices, value: Any) -> None:
        # TODO: optimize this method to avoid resolving history if location
        # is slice(None) or Ellipsis and history only has one element
        # in this case, only previous_array is required
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
                    location = shift_indices(location, self.shift, self.apparent_size)
                destination = self._previous_array
            else:
                destination = self._current_array
        else:
            location = shift_indices(location, self.shift, self.apparent_size)
            destination = self._base_array
        destination[location] = value
    
    def rotate(self) -> None:

        if self._index_history:
            raise RuntimeError("Only allowed to call `rotate()` on original array")

        self.offset += self.apparent_size

        if self._padding and self.offset >= self.full_size:
            # copy values from end of base array to beginning
            final_values = slice(-(self._padding * 2), None)
            next_previous_values = slice(0, self._padding * 2)
            self._base_array[next_previous_values] = self._base_array[final_values]

        self.offset %= self.full_size

        # current array is now invalid, but apparent shape should still be
        # correct
        self._current_array = None

    @property
    def full(self) -> LargeArray:
        full: LargeArray = self.__new__(type(self))
        full.__dict__.update(self.__dict__)

        full.apparent_size = full.full_size
        full.offset = 0

        full._index_history = []
        full._current_array = None
        full._apparent_shape = None
        return full

    @property
    def next(self) -> LargeArray:
        return self._get_at_offset(offset=1)

    @property
    def previous(self) -> LargeArray:
        return self._get_at_offset(offset=-1)

    def _get_at_offset(self, offset: int) -> LargeArray:
        if self._index_history:
            raise RuntimeError("Only allowed to get at offset from unindexed array")

        new: LargeArray = self.__new__(type(self))
        new.__dict__.update(self.__dict__)

        # total shift of offset cannot exceed +/-padding, but we do not check
        # if padding is exceeded, getitem/setitem may throw error
        new.offset += offset

        # index_history is already empty
        # current array is now invalid, but apparent shape should still be
        # correct
        new._current_array = None
        return new


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
