import itertools
from typing import Any, Optional

import numpy as np

from parllel.arrays.array import Array
from parllel.buffers import Buffer, Index, Indices

from .array import Array
from .indices import add_locations, add_indices, shape_from_indices


class JaggedArray(Array):
    """
    
    # TODO: ensure that index_lib can handle np_arrays as indices
    # TODO: maybe create a repr that doesn't call __array__, as this can be expensive
    """
    def __init__(self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        *,
        batch_size: tuple[int, ...],
        storage: str = "local",
        padding: int = 0,
        full_size: Optional[int] = None,
    ) -> None:

        if batch_size == ():
            batch_size = (1,)
        self.batch_size = batch_size

        dtype = np.dtype(dtype)
        if dtype == np.object_:
            raise ValueError("Data type should not be object.")
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

        self._current_array = None

        # TODO: move this into allocate()
        # TODO: add final element along T dimension for signaling end of data
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
        # TODO: if accessed location has not been writted to yet, throw an error
        
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

            # TODO: if the minimum value of ptrs[1:] is not 0, the array is full
            # handle this either by dropping input, wrapping, or extending array

            if cursor != t_loc:
                raise IndexError(
                    "JaggedArray requires consecutive writes. The next "
                    f"available slot is at {cursor}, not {t_loc}."
                )

            # write to that location
            ptrs[t_loc + 1] = ptrs[t_loc] + value.shape[0]

            n_slice = slice(ptrs[t_loc], ptrs[t_loc + 1])
            n_loc = add_indices(n_slice, feature_locs[0])

            real_loc = batch_loc + (n_loc,) + feature_locs[1:]

            self._base_array[real_loc] = value

    def __array__(self, dtype=None) -> np.ndarray:
        if self._apparent_shape is None:
            self._resolve_indexing_history()
        
        current_location = tuple(self._current_location)
        batch_locs, feature_locs = current_location[:len(self.batch_size)], current_location[len(self.batch_size):]
        t_locs, b_locs = batch_locs[0], batch_locs[1:]

        # create a list of graphs to be concatenated into one large "batch"
        graphs = []
        current_ptrs = []

        # loop over all batch locations except the T dimension
        b_locs = (
            (loc,) if isinstance(loc, int) else slice_to_list(loc, size)
            for loc, size
            in zip(b_locs, self._apparent_base_shape)
        )
        for b_loc in itertools.product(*b_locs):
            if isinstance(t_locs, int):
                ptrs = self._ptr[(slice(None),) + b_loc]
                n_slice = slice(ptrs[t_locs], ptrs[t_locs + 1])
                n_loc = add_indices(n_slice, feature_locs[0])
                real_loc = b_loc + (n_loc,) + feature_locs[1:]
                graph = self._base_array[real_loc]
                graphs.append(graph)
                current_ptrs.append(graph.shape[0])

            else:  # t_locs: slice
                raise NotImplementedError

        array = np.concatenate(graphs) if len(graphs) > 1 else graphs[0]
        current_ptrs = np.cumsum(current_ptrs)
        assert array.shape[0] == current_ptrs[-1]
        current_ptrs[1:] = current_ptrs[:-1]
        current_ptrs[0] = 0

        if dtype is not None:
            array = array.astype(dtype, copy=False)

        self._current_ptrs = current_ptrs  # save for consumption by __buffer__
        return array

    def __buffer__(self) -> Buffer:
        data = self.__array__()
        return (
            data,
            self._current_ptrs,  # updated during execution of __array__
        )


def slice_to_list(slice_: slice, size: int) -> list[int]:
    start, stop, step = slice_.indices(size)
    return list(range(start, stop, step))
