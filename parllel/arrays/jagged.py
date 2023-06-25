from __future__ import annotations
import itertools
from typing import Literal, Optional, Union

import numpy as np

from parllel.arrays.array import Array
from parllel.buffers import Buffer, Index, Indices
import parllel.logger as logger

from .array import Array
from .indices import add_locations, add_indices, shape_from_indices


class JaggedArray(Array):
    """
    
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
        on_overflow: Literal["drop", "resize", "wrap"] = "drop",
    ) -> None:
        dtype = np.dtype(dtype)
        if dtype == np.object_:
            raise ValueError("Data type should not be object.")

        if batch_size == ():
            batch_size = (1,)

        if on_overflow not in {"drop", "resize", "wrap"}:
            raise ValueError(f"Unknown on_overflow option {on_overflow}")
        elif on_overflow in {"resize", "wrap"}:
            raise NotImplementedError(f"on_overflow option {on_overflow} not permitted")
        
        self.dtype = dtype
        # self.batch_size = batch_size
        self.padding = padding
        self.on_overflow = on_overflow

        self.offset = padding
        self.base_size = batch_size[0]

        T_size = batch_size[0]
        N_size = shape[0]
        # multiply the T dimension into the node dimension
        self._flattened_size = T_size * N_size
        self._base_shape = batch_size[1:] + (self._flattened_size,) + shape[1:]
        self._n_batch_dim = len(batch_size)

        self._apparent_base_shape = self._apparent_shape = batch_size + shape

        self._buffer_id: int = id(self)
        self._index_history: list[Indices] = []
        self._current_location: list[Index] = [slice(None) for _ in self._apparent_base_shape]

        self._allocate()

        self._current_array = None

        # TODO: move this into allocate()
        # add an extra element to node dimension so it's always possible to
        # access the element at t+1
        self._ptr = np.zeros(shape=batch_size[1:] + (T_size + 1,), dtype=np.int64)

        self._resolve_indexing_history()

    def _resolve_indexing_history(self) -> None:
        for location in self._index_history:
            self._current_location = add_locations(self._current_location, location)
        
        self._index_history.clear()

        self._apparent_shape = shape_from_indices(self._apparent_base_shape,
                                                  self._current_location)
        
        # TODO: if current location is a single graph, set real size of that graph
        # as apparent size
        
    def __setitem__(self, location: Indices, value: Union[float, int, np.number, list, JaggedArray]) -> None:
        if self._apparent_shape is None:
            self._resolve_indexing_history()

        destination = tuple(add_locations(self._current_location, location))

        if isinstance(value, list):
            # TODO: zip and iterate over value and location
            raise NotImplementedError("Cannot write a list into a JaggedArray.")
        elif isinstance(value, JaggedArray):
            raise NotImplementedError("Cannot write a JaggedArray into a JaggedArray.")
        elif isinstance(value, (int, float)) and value == 0:
            # TODO: test this method of element deletion
            value = np.array([])  # special value for deleting an entry
        value = np.atleast_1d(value)  # otherwise promote scalars to 1D arrays

        # split into batch locations and feature locations, which are handled differently
        batch_locs, feature_locs = destination[:self._n_batch_dim], destination[self._n_batch_dim:]
        
        # loop over all batch locations by taking the product of slices
        batch_locs = (
            (loc,) if isinstance(loc, int) else slice_to_range(loc, size)
            for loc, size
            in zip(batch_locs, self._apparent_base_shape)
        )
        for loc in itertools.product(*batch_locs):

            t_loc, batch_loc = loc[0], loc[1:]

            # get lengths of all graphs stored in this "element"
            ptrs = self._ptr[batch_loc]

            if (end := ptrs[t_loc] + value.shape[0]) > self._flattened_size:
                if self.on_overflow == "resize":
                    # TODO: add a resize method and call it here
                    raise NotImplementedError
                else:
                    # drop input
                    logger.debug(f"JaggedArray input {value} dropped due to size exceeded.")
                    return

            # set end point at ptrs[t_loc + 1]
            ptrs[t_loc + 1] = end

            # to ensure that ptrs remains monotonic, remove end points of entries
            # that have now been overwritten
            ptrs_after = ptrs[t_loc + 1:]
            ptrs_after[ptrs_after < end] = end
            # TODO: to prevent fragments of overwritten entries from remaining,
            # make ptrs actual two values (start and end) for each entry (add
            # trailing dimension of size 2)

            # compute the subarray within the slice of all nodes
            n_slice = slice(ptrs[t_loc], ptrs[t_loc + 1])
            n_loc = add_indices(n_slice, feature_locs[0])
            real_loc = batch_loc + (n_loc,) + feature_locs[1:]

            # write to that location
            self._base_array[real_loc] = value

    def __array__(self, dtype=None) -> np.ndarray:
        if self._apparent_shape is None:
            self._resolve_indexing_history()
        
        current_location = tuple(self._current_location)
        batch_locs, feature_locs = current_location[:self._n_batch_dim], current_location[self._n_batch_dim:]
        t_locs, b_locs = batch_locs[0], batch_locs[1:]

        # create a list of graphs to be concatenated into one large "batch"
        graphs = []
        current_ptrs = []

        # loop over all batch locations except the T dimension
        b_locs = (
            (loc,) if isinstance(loc, int) else slice_to_range(loc, size)
            for loc, size
            in zip(b_locs, self._apparent_base_shape)
        )
        for b_loc in itertools.product(*b_locs):
            ptrs = self._ptr[b_loc]
            if isinstance(t_locs, int):
                n_slice = slice(ptrs[t_locs], ptrs[t_locs + 1])
                n_loc = add_indices(n_slice, feature_locs[0])
                real_loc = b_loc + (n_loc,) + feature_locs[1:]
                graph = self._base_array[real_loc]
                graphs.append(graph)
                current_ptrs.append(graph.shape[0])

            # t_locs: slice
            elif ((n_locs := feature_locs[0]) == slice(None) and 
                  np.abs(t_locs.step) == 1):
                # TODO: verify this works for step == -1
                start, stop, _ = t_locs.indices(ptrs.shape[0])
                n_loc = slice(ptrs[start], ptrs[stop + 1])
                real_loc = b_loc + (n_loc,) + feature_locs[1:]
                graph = self._base_array[real_loc]
                graphs.append(graph)
                current_ptrs.append(graph.shape[0])

            else:
                # TODO: iterate over elements of t_locs and index each one
                # using n_locs
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


def slice_to_range(slice_: slice, size: int) -> list[int]:
    start, stop, step = slice_.indices(size)
    stop = None if stop < 0 else stop
    return range(start, stop, step)
