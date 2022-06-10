from typing import Any, Tuple

import numpy as np

from parllel.buffers import Indices

from .array import Array
from .indices import (compute_indices, does_index_scalar, slicify_final_index,
    shift_indices)


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

    @property
    def current_indices(self):
        if self._current_indices is None:

            self._resolve_indexing_history()

            self._current_indices = compute_indices(
                self._base_array, self._current_array)

            self._current_indices = shift_indices(self._current_indices, -self._padding)

        return self._current_indices

    def _resolve_indexing_history(self) -> None:
        if self._index_history:
            return super()._resolve_indexing_history()

        if self._unresolved_indices:
            location = self._unresolved_indices.pop(0)
            self._index_history.append(location)

            # shift the indices to account for padding
            location = shift_indices(location, self._padding)

            if does_index_scalar(self._apparent_shape, location):
                # sneakily turn final index into a slice so that indexing a
                # scalar (which results in a copy) is avoided
                location = slicify_final_index(location)                
                self._current_array = self._base_array[location]
                self._apparent_shape = ()

            else:
                self._current_array = self._base_array[location]
                self._apparent_shape = self._current_array.shape

        else:
            # even if the array was never indexed, only this slice of the array
            # should be returned by __array__
            location = slice(self._padding, -self._padding)
            self._current_array = self._base_array[location]
            self._apparent_shape = self._current_array.shape

        super()._resolve_indexing_history()

    def __setitem__(self, location: Indices, value: Any) -> None:
        if self.index_history:
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
        if not self.index_history:
            # only rotate if called on the base array.
            # rotating subarrays is not possible anyway
            final_values = slice(-(self._padding * 2), None)
            next_previous_values = slice(0, self._padding * 2)
            self._base_array[next_previous_values] = self._base_array[final_values]
