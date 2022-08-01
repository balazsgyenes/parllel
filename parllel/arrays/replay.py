from functools import reduce
from typing import Any, Tuple

import numpy as np

from parllel.buffers import Index, Indices

from .array import Array


class ReplayArray(Array):
    def __init__(self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        replay_length: int = 1,
    ) -> None:
        """TODO: add padding?

        Problems with this class:
        - in the sampler init, we need to set up initial values without calling
            rotate, because that changes state now. can write to observation[0],
            but how to write to values[-1] only if value is a RotatingArray?
        - when replay length is not a multiple of shape[0], we may have to
            provide a view on disjoint sections of the array
        """

        if not replay_length > shape[0]:
            raise ValueError("Replay length must be at least the apparent"
                " leading dimension of the array.")
        self._replay_length = replay_length

        if not shape:
            raise ValueError("Non-empty shape required.")

        # add padding onto both ends of first dimension
        padded_shape = (shape[0] + 2 * self._padding,) + shape[1:]

        super().__init__(shape=padded_shape, dtype=dtype)

        # unlike in base class, _current_array is not the same as _base_array
        # this also fixes _apparent_shape, which still includes the padding
        self._resolve_indexing_history()

    # @property
    # def padding(self) -> int:
    #     return self._padding

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
        array = self._base_array

        if self._index_history:
            # shift only the first indices, leave the rest (if thereare more)
            index_history = [shift_indices(self._index_history[0], self._padding),
                            ] + self._index_history[1:]
        else:
            # even if the array was never indexed, only this slice of the array
            # should be returned by __array__
            index_history = [slice(self._padding, -self._padding)]

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
                # in this case, there must be an index history
                location = self._index_history[-1]
                # indices must be shifted if they were the first indices
                if len(self._index_history) == 1:
                    location = shift_indices(location, self._padding)
                destination = self._previous_array
            else:
                destination = self._current_array
        else:
            location = shift_indices(location, self._padding)
            destination = self._base_array
        destination[location] = value

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
