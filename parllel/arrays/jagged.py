from __future__ import annotations

import itertools
from typing import Literal, Optional, Union

import numpy as np

import parllel.logger as logger
from parllel.arrays.array import Array
from parllel.buffers import Buffer, Index, Indices

from .array import Array
from .indices import (add_locations, index_slice, init_location,
                      shape_from_location)


class JaggedArray(Array, kind="jagged"):
    """An array that represents a list of arrays, the sizes of which may differ
    in their leading dimension.
    """
    # TODO: maybe create a repr that doesn't call __array__, as this can be expensive
    # TODO: ensure consistent init args between this and Array
    # TODO: ensure that shape and batch_shape are stored
    kind = "jagged"

    def __init__(self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        *,
        kind: Optional[str] = None,
        storage: Optional[str] = None,
        batch_shape: tuple[int, ...],
        padding: int = 0,
        full_size: Optional[int] = None,
        on_overflow: Literal["drop", "resize", "wrap"] = "drop",
    ) -> None:
        dtype = np.dtype(dtype)
        if dtype == np.object_:
            raise ValueError("Data type should not be object.")

        if batch_shape == ():
            batch_shape = (1,)

        if padding < 0:
            raise ValueError("Padding must be non-negative.")
        if padding > shape[0]:
            raise ValueError(
                f"Padding ({padding}) cannot be greater than leading "
                f"dimension {shape[0]}."
            )

        if on_overflow not in {"drop", "resize", "wrap"}:
            raise ValueError(f"Unknown on_overflow option {on_overflow}")
        elif on_overflow in {"resize", "wrap"}:
            raise NotImplementedError(f"{on_overflow=}")
        
        if full_size is None:
            full_size = batch_shape[0]
        else:
            raise NotImplementedError(f"{full_size=}")
        if full_size < batch_shape[0]:
            raise ValueError(
                f"Full size ({full_size}) cannot be less than "
                f"leading batch dimension {batch_shape[0]}."
            )
        if full_size % batch_shape[0] != 0:
            raise ValueError(
                f"The leading dimension {batch_shape[0]} must divide the full "
                f"size ({full_size}) evenly."
            )

        self.dtype = dtype
        self.padding = padding

        base_T_size = full_size + 2 * padding
        N_size = shape[0]
        self._default_size = batch_shape[0]  # size of leading dim of unindexed array
        self._virtual_base_shape = (base_T_size,) + batch_shape[1:] + shape  # shape that base array appears to be
        self._full_size = full_size  # size of leading dim of full array, without padding
        self._offset = 0  # offset of visible region within full array, without padding
        self._shift = self._offset + padding  # offset of visible region within base array
        # multiply the T dimension into the node dimension
        self._flattened_size = base_T_size * N_size
        self._base_shape = batch_shape[1:] + (self._flattened_size,) + shape[1:]
        self._n_batch_dim = len(batch_shape)

        self.on_overflow = on_overflow
        
        self._buffer_id: int = id(self)
        self._index_history: list[Indices] = []
        self._current_location = init_location(self._virtual_base_shape)
        if padding:
            init_slice = slice(padding, self._default_size + padding)
            self._current_location = add_locations(
                self._current_location, init_slice, self._virtual_base_shape,
            )

        self._allocate()
        # TODO: move this into allocate()
        # add an extra element to node dimension so it's always possible to
        # access the element at t+1
        self._ptr = np.zeros(shape=batch_shape[1:] + (base_T_size + 1,), dtype=np.int64)

        self._current_array = None  # not used by JaggedArray
        self._apparent_shape = batch_shape + shape  # shape of current array

    def _resolve_indexing_history(self) -> None:
        for location in self._index_history:
            self._current_location = add_locations(
                self._current_location,
                location,
                self._virtual_base_shape,
                neg_from_end=False,
            )
        
        self._index_history.clear()

        self._apparent_shape = shape_from_location(self._current_location, self._virtual_base_shape)
        
        # TODO: if current location is a single graph, set real size of that graph
        # as apparent size
        
    def __setitem__(self, location: Indices, value: Union[float, int, np.number, list, JaggedArray]) -> None:
        if self._apparent_shape is None:
            self._resolve_indexing_history()

        destination = tuple(add_locations(
            self._current_location,
            location,
            self._virtual_base_shape,
            neg_from_end=False,
        ))

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
            (loc,) if isinstance(loc, int) else slice_to_range(loc)
            for loc in batch_locs
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
            start, end = ptrs[t_loc], ptrs[t_loc + 1]
            n_slice = slice(start, end, 1)  # standard slice must have integer step
            n_loc = index_slice(n_slice, feature_locs[0], self._flattened_size)
            real_loc = batch_loc + (n_loc,) + feature_locs[1:]

            # write to that location
            self._base_array[real_loc] = value

    @property
    def full(self) -> JaggedArray:
        full: JaggedArray = self.__new__(type(self))
        full.__dict__.update(self.__dict__)

        full._default_size = full._full_size
        full._offset = 0

        full._index_history = []
        full._current_array = None
        full._apparent_shape = None

        # clear current location
        # TODO: this can be deleted once Array has been refactored
        full._current_location = init_location(full._virtual_base_shape)
        if full.padding:
            init_slice = slice(full.padding, full._default_size + full.padding)
            full._current_location = add_locations(
                full._current_location, init_slice, full._virtual_base_shape,
            )

        return full

    def rotate(self) -> None:
        # TODO: once setiem supports JaggedArray, no override needed
        self._offset += self._default_size

        if self.padding and self._offset >= self._full_size:
            # copy values from end of base array to beginning
            final_values = range(
                self._full_size - self.padding,
                self._full_size + self.padding,
            )
            next_previous_values = range(-self.padding, self.padding)
            b_locs = [range(size) for size in self._virtual_base_shape[1:self._n_batch_dim]]
            for source, destination in zip(final_values, next_previous_values):
                for b_loc in itertools.product(*b_locs):
                    self.full[(destination,) + b_loc] = np.asarray(self.full[(source,) + b_loc])

        self._offset %= self._full_size

    def __array__(self, dtype=None) -> np.ndarray:
        if self._apparent_shape is None:
            self._resolve_indexing_history()
        
        current_location = tuple(self._current_location)
        batch_locs, feature_locs = current_location[:self._n_batch_dim], current_location[self._n_batch_dim:]
        
        if any(isinstance(loc, np.ndarray) for loc in feature_locs):
            raise IndexError("No advanced indexing permitted within feature dimensions.")

        # create a list of graphs to be concatenated into one large "batch"
        graphs = []
        current_ptrs = []

        if any(isinstance(loc, np.ndarray) for loc in batch_locs):
            # advanced indexing: zip arrays and iterate over them together
            # TODO: also iterate over product of this zip any slices present
            batch_locs = zip(
                *(
                    iter(loc) if isinstance(loc, np.ndarray) else (loc,)
                    for loc in batch_locs
                )
            )
        else:
            # iterate over every combination of batch indices, treating every
            # graph individually
            # TODO: can optimize by handling slices in T dim separately, as
            # these are contiguous array regions
            batch_locs = itertools.product(
                *(
                    (loc,) if isinstance(loc, int) else slice_to_range(loc)
                    for loc in batch_locs
                )
            )

        # loop over all batch locations except the T dimension
        for batch_loc in batch_locs:
            if any(isinstance(loc, slice) for loc in batch_loc):
                raise NotImplementedError(f"{batch_loc=}")

            t_loc, b_loc = batch_loc[0], batch_loc[1:]
            ptrs = self._ptr[b_loc]

            start, end = ptrs[t_loc], ptrs[t_loc + 1]
            n_slice = slice(start, end, 1)
            n_loc = index_slice(n_slice, feature_locs[0], self._flattened_size)
            real_loc = b_loc + (n_loc,) + feature_locs[1:]
            graph = self._base_array[real_loc]
            graphs.append(graph)
            current_ptrs.append(graph.shape[0])

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


def slice_to_range(slice_: slice) -> list[int]:
    start, stop, step = slice_.start, slice_.stop, slice_.step
    stop = -1 if stop is None else stop
    return range(start, stop, step)
