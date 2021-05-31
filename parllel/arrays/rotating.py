from __future__ import annotations
import copy
from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray

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

        super().__init__(padded_shape, dtype)
        self._apparent_shape = shape # padding is not part of apparent shape

        self._apparent_index_history: List[Indices] = []

    @Buffer.index_history.getter
    def index_history(self) -> Tuple[Tuple[Index, ...], ...]:
        """For external use, return the index history before shifting. The base
        class index history records indices after shifting.
        """
        return self._apparent_index_history

    @property
    def end(self) -> int:
        """The index of the final element in the array, not including padding.
        """
        return self._apparent_shape[0] - 1

    def initialize(self) -> None:
        super().initialize()

        self._hidden_array: NDArray = np.zeros(shape=self.shape[1:], dtype=self.dtype)
        self._hidden_index = None
        self._show_hidden_proxy = ShowHiddenProxyArray(self)

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
        result._padding = 0 # padding is no longer available after indexing
        # assign copy of _index_history with additional element for this
        # indexing operation
        result._apparent_index_history = copy.copy(
            result._apparent_index_history) + [location]
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

    @property
    def temporary(self):
        return self._show_hidden_proxy

    def __array__(self, dtype = None) -> NDArray:
        array = super().__array__(dtype)
        if self._padding > 0:
            array = array[self._padding:-self._padding]
        return array


class ShowHiddenProxyArray:
    def __init__(self, parent: RotatingArray):
        self._parent = parent

    def __setitem__(self, location: Any, value: Any) -> None:
        parent: RotatingArray = self._parent
        if not isinstance(location, tuple):
            location = (location, )
        parent._hidden_index = location[0]
        assert isinstance(parent._hidden_index, int), (
            "When assigning to temporary array, leading index must be an "
            "integer."
        )
        parent._hidden_array[location[1:]] = value


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
