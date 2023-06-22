import itertools
from typing import Any, Optional

import numpy as np

from parllel.arrays.array import Array
from parllel.buffers import Index, Indices

from .array import Array
from .indices import add_locations, add_indices, shape_from_indices


class JaggedArray(Array):
    def __init__(self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        batch_size: tuple[int, ...],
        *,
        storage: str = "local",
        padding: int = 0,
        full_size: Optional[int] = None,
    ) -> None:

        if batch_size == ():
            batch_size = (1,)
        self.batch_size = batch_size

        self.dtype = dtype
        self.padding = padding
        self.offset = padding

        self.base_size = batch_size[0]

        # multiply the T dimension into the node dimension
        self._base_shape = batch_size[1:] + (batch_size[0] * shape[0],) + shape[1:]

        self._apparent_base_shape = self._apparent_shape = batch_size + shape

        self._buffer_id: int = id(self)
        self._index_history: list[Indices] = []
        self._current_location: list[Index] = [slice(None) for _ in self._apparent_base_shape]

        self._allocate()

        # TODO: move this into allocate()
        self._ptr = np.zeros(shape=batch_size, dtype=np.int64)

        self._resolve_indexing_history()

    def _resolve_indexing_history(self) -> None:
        for location in self._index_history:
            self._current_location = add_locations(self._current_location, location)
        
        self._index_history.clear()

        self._apparent_shape = shape_from_indices(self._apparent_base_shape,
                                                  self._current_location)
        
        # TODO: if current location is a single graph, set real size of that graph
        # as apparent size
        
    def __setitem__(self, location: Indices, value: Any) -> None:
        # TODO: allow writes to previously written locations
        # TODO: zip and iterate over value and location
        # TODO: value can be a scalar (python or numpy array), a list, or a JaggedArray

        if self._apparent_shape is None:
            self._resolve_indexing_history()

        destination = tuple(add_locations(self._current_location, location))

        # split into batch locations and feature locations, which are handled differently
        batch_locs, feature_locs = destination[:len(self.batch_size)], destination[len(self.batch_size):]
        
        # loop over all batch locations
        batch_locs = (
            (loc,) if isinstance(loc, int) else slice_to_list(loc, size)
            for loc, size
            in zip(batch_locs, self._apparent_base_shape)
        )
        for loc in itertools.product(*batch_locs):

            t_loc, batch_loc = loc[0], loc[1:]

            # get lengths of all graphs stored in this "element"
            ptrs = self._ptr[(slice(None),) + batch_loc]

            # find next available slot
            cursor = np.argmin(ptrs[1:])

            if cursor != t_loc:
                raise IndexError(
                    "JaggedArray requires consecutive writes. The next "
                    f"available slot is at {cursor}, not {t_loc}."
                )
            cursor += 1

            # write to that location
            ptrs[cursor] = ptrs[cursor - 1] + value.shape[0]

            n_slice = slice(ptrs[cursor - 1], ptrs[cursor])
            n_loc = add_indices(n_slice, feature_locs[0])

            real_loc = batch_loc + (n_loc,) + feature_locs[1:]

            self._base_array[real_loc] = value

    def __array__(self, dtype=None) -> np.ndarray:
        if self._apparent_shape is None:
            self._resolve_indexing_history()
        
        (t_loc, *batch_loc), feature_loc = self._current_location[:len(self.batch_size)], self._current_location[len(self.batch_size):]

        # lengths = self._cum_lengths[:, batch_loc]

        if isinstance(t_loc, int):
            end = self._cum_lengths[t_loc, batch_loc]
            start = self._cum_lengths[t_loc - 1, batch_loc] if t_loc > 0 else 0

            # TODO: check that end > 0 to ensure that the desired graph has been written

            # get indices of nodes within desired graph
            n_index = (
                self._current_location[len(self.batch_size)]
                if len(self._current_location) > len(self.batch_size) else
                slice(None)
            )

            # convert to global node index across all graphs
            n_index = add_indices(slice(start, end), n_index)

            array = self._base_array
            array = np.asarray(array)  # promote scalars to 0d arrays
        else:
            pass

        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def __buffer__(self) -> dict:
        data = self.__array__()


def slice_to_list(slice_: slice, size: int) -> list[int]:
    start, stop, step = slice_.indices(size)
    return list(range(start, stop, step))
