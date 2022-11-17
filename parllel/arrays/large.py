from functools import reduce
from typing import Any, Optional, Tuple

import numpy as np

from parllel.buffers import Index, Indices

from .array import Array
from .rotating import RotatingArray, shift_indices


class LargeArray(RotatingArray):

    def __init__(self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        padding: int = 0,
        apparent_leading_dim: Optional[int] = None,
    ) -> None:

        if apparent_leading_dim is None:
            apparent_leading_dim = shape[0]
        if shape[0] % apparent_leading_dim != 0:
            raise ValueError(
                "The leading dimension of the array must divide evenly into "
                "the apparent leading dimension."
            )
        self.apparent_leading_dim = apparent_leading_dim
        self.large = (shape[0] > apparent_leading_dim)

        super().__init__(shape, dtype, padding)

        self.offset = 0

    def _resolve_indexing_history(self) -> None:
        array = self._base_array
        shift = self.offset + self._padding

        if self._index_history:
            # shift only the first indices, leave the rest (if thereare more)
            # TODO: modify this method so it also takes the apparent shape
            # right now, it won't account for the hidden region
            index_history = [shift_indices(self._index_history[0], shift),
                            ] + self._index_history[1:]
        elif self.padding > 0 or self.large:
            # even if the array was never indexed, only this slice of the array
            # should be returned by __array__
            index_history = [slice(shift, shift + self.apparent_leading_dim)]
        else:
            # never indexed and no padding
            index_history = [slice(None)]

        # if index history has only 1 element, this has no effect
        array = reduce(lambda arr, index: arr[index], index_history[:-1], array)
        self._previous_array = array
        
        # we guarantee that index_history has at least 1 element
        array = array[index_history[-1]]

        self._current_array = array
        self._apparent_shape = array.shape
    
    def rotate(self) -> None:

        self.offset += self.apparent_leading_dim

        super().rotate()
