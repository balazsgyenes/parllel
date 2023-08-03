# fmt: off
from __future__ import annotations

import itertools
from typing import Any, Literal, TypeVar

import numpy as np

import parllel.logger as logger
from parllel import ArrayDict
from parllel.arrays.array import Array
from parllel.arrays.indices import (Location, StandardIndex,
                                    batch_dims_from_location, compose_indices,
                                    compose_locations, init_location)
from parllel.arrays.managedmemory import SharedMemoryArray
from parllel.arrays.sharedmemory import InheritedMemoryArray

# fmt: on

Self = TypeVar("Self", bound="JaggedArray")


class JaggedArray(Array, kind="jagged"):
    """An array that represents a list of arrays, the sizes of which may differ
    in their leading dimension.
    """

    # TODO: maybe create a repr that doesn't call __array__, as this can be expensive
    kind = "jagged"
    _base_array: np.ndarray
    _ptr: np.ndarray

    @classmethod
    def _specialize_subclass(
        cls,
        *args,
        batch_shape: tuple[int, ...] | None = None,
        full_size: int | None = None,
        **kwargs,
    ) -> type[Array]:
        if full_size is None or full_size == batch_shape[0]:
            # batch_shape is only None if calling JaggedArray[List].__new__(type(self))
            # explicitly: must return the same type in order for pickling to work
            return cls
        else:
            from parllel.arrays.jagged_list import JaggedArrayList

            return JaggedArrayList

    def __init__(
        self,
        batch_shape: tuple[int, ...],
        dtype: np.dtype,
        *,
        max_mean_num_elem: int,
        feature_shape: tuple[int, ...] = (),
        kind: str | None = None,
        storage: str | None = None,
        padding: int = 0,
        full_size: int | None = None,
        on_overflow: Literal["drop", "resize", "wrap"] = "drop",
    ) -> None:
        if not batch_shape:
            raise ValueError("Non-empty batch_shape required.")

        dtype = np.dtype(dtype)
        if dtype == np.object_:
            raise ValueError("Data type should not be object.")

        if padding < 0:
            raise ValueError("Padding must be non-negative.")
        if padding > batch_shape[0]:
            raise ValueError(
                f"Padding ({padding}) cannot be greater than leading dimension {batch_shape[0]}."
            )

        if full_size is None:
            full_size = batch_shape[0]
        if full_size < batch_shape[0]:
            raise ValueError(
                f"Full size ({full_size}) cannot be less than leading dimension {batch_shape[0]}."
            )
        if full_size % batch_shape[0] != 0:
            raise ValueError(
                f"Full size ({full_size}) must be evenly divided by leading dimension {batch_shape[0]}."
            )

        if on_overflow not in {"drop", "resize", "wrap"}:
            raise ValueError(f"Unknown on_overflow option {on_overflow}")
        elif on_overflow in {"resize", "wrap"}:
            raise NotImplementedError(f"{on_overflow=}")

        self.dtype = dtype
        self.padding = padding
        self.on_overflow = on_overflow
        self.max_mean_num_elem = max_mean_num_elem
        # size of leading dim of full array, without padding
        self.full_size = full_size
        # shape that base array appears to be
        self._base_shape = (
            (full_size + 2 * padding,)
            + batch_shape[1:]
            + (max_mean_num_elem,)
            + feature_shape
        )
        self._base_batch_dims = len(batch_shape)
        # multiply the T dimension into the node dimension
        self._flattened_size = self._base_shape[0] * max_mean_num_elem

        allocate_shape = batch_shape[1:] + (self._flattened_size,) + feature_shape
        self._allocate(shape=allocate_shape, dtype=self.dtype, name="_base_array")

        # add an extra element to node dimension so it's always possible to
        # access the element at t+1
        ptr_shape = batch_shape[1:] + (self._base_shape[0] + 1,)
        self._allocate(shape=ptr_shape, dtype=np.int64, name="_ptr")

        self._current_location = init_location(self._base_shape)
        init_slice = slice(padding, batch_shape[0] + padding)
        self._unresolved_indices: list[Location] = [init_slice]
        self._resolve_indexing_history()

        self._rotatable = True

    def new_array(self, *args, **kwargs) -> Array:
        """Creates an Array with the same shape and type as a given Array
        (similar to torch's new_zeros function). By default, the full size of
        the created Array is just the apparent size of the template. To set it
        to another value, either pass it manually or set the
        `inherit_full_size` flag to True to use the template's full size.
        """
        if "kind" not in kwargs or kwargs["kind"] == "jagged":
            max_mean_num_elem: int | None = kwargs.get("max_mean_num_elem")
            kwargs["max_mean_num_elem"] = (
                max_mean_num_elem if max_mean_num_elem else self.max_mean_num_elem
            )

            feature_shape: tuple[int, ...] | None = kwargs.get("feature_shape")
            if feature_shape is None:
                # get number of visible dimensions that are not feature dimensions
                n_batch_and_point_dims = batch_dims_from_location(
                    self._current_location,
                    self._base_batch_dims + 1,
                )
                feature_shape = self.shape[n_batch_and_point_dims:]
            kwargs["feature_shape"] = feature_shape

            on_overflow: str | None = kwargs.get("on_overflow")
            kwargs["on_overflow"] = on_overflow if on_overflow else self.on_overflow
        return super().new_array(*args, **kwargs)

    @classmethod
    def _get_from_numpy_kwargs(cls, example: Any, kwargs: dict) -> dict:
        if kwargs.get("feature_shape") is None:
            # promote scalars to 0d arrays
            np_example: np.ndarray = np.asanyarray(example)
            if len(np_example.shape) == 0:
                raise ValueError(
                    "Expected example of data with variable-sized leading dimension."
                )
            kwargs["feature_shape"] = np_example.shape[1:]
        return super()._get_from_numpy_kwargs(example[0], kwargs)

    def _resolve_indexing_history(self) -> None:
        super()._resolve_indexing_history()

        # TODO: if current location is a single graph, set real size of that graph
        # as apparent size

    def __setitem__(
        self,
        indices: Location,
        value: float | int | np.number | list | JaggedArray,
    ) -> None:
        if self._shape is None:
            self._resolve_indexing_history()

        destination = tuple(
            compose_locations(
                self._current_location,
                indices,
                self._base_shape,
                neg_from_end=False,
            )
        )

        if isinstance(value, list):
            # TODO: zip and iterate over value and location
            raise NotImplementedError("Cannot write a list into a JaggedArray.")
        elif isinstance(value, JaggedArray):
            raise NotImplementedError("Cannot write a JaggedArray into a JaggedArray.")
        elif np.isscalar(value) and value == 0:
            # TODO: test this method of element deletion
            value_n_points = 0  # delete by writing an element of 0 size
        else:
            value_n_points = value.shape[0]
        value = np.atleast_1d(value)  # promote scalars to 1D arrays

        # split into batch locations and feature locations, which are handled differently
        batch_locs, feature_locs = (
            destination[: self._base_batch_dims],
            destination[self._base_batch_dims :],
        )

        # loop over all batch locations by taking the product of slices
        batch_locs = (
            (loc,) if isinstance(loc, int) else slice_to_range(loc)
            for loc in batch_locs
        )
        for loc in itertools.product(*batch_locs):
            t_loc, batch_loc = loc[0], loc[1:]

            # get lengths of all graphs stored in this "element"
            ptrs = self._ptr[batch_loc]

            if (end := ptrs[t_loc] + value_n_points) > self._flattened_size:
                if self.on_overflow == "resize":
                    # TODO: add a resize method and call it here
                    raise NotImplementedError
                else:
                    # drop input
                    logger.warn(
                        f"JaggedArray input {value} dropped due to size exceeded."
                    )
                    return

            # set end point at ptrs[t_loc + 1]
            ptrs[t_loc + 1] = end

            # to ensure that ptrs remains monotonic, remove end points of entries
            # that have now been overwritten
            ptrs_after = ptrs[t_loc + 1 :]
            ptrs_after[ptrs_after < end] = end
            # TODO: to prevent fragments of overwritten entries from remaining,
            # make ptrs actual two values (start and end) for each entry (add
            # trailing dimension of size 2)

            # compute the subarray within the slice of all nodes
            start, end = ptrs[t_loc], ptrs[t_loc + 1]
            n_slice = slice(start, end, 1)  # standard slice must have integer step
            # use neg_from_end=True here to ensure that slice doesn't get
            # extended beyond the size of the point cloud
            n_loc = compose_indices(n_slice, feature_locs[0], self._flattened_size)
            real_loc = batch_loc + (n_loc,) + feature_locs[1:]

            # write to underlying array at that location
            self._base_array[real_loc] = value

    def rotate(self) -> None:
        # TODO: once setitem supports JaggedArray, no override needed
        if not self._rotatable:
            raise ValueError("rotate() is only allowed on unindexed array.")

        leading_loc: slice = self._current_location[0]
        start = leading_loc.start + self.shape[0]
        stop = leading_loc.stop + self.shape[0]

        if start >= self.full_size:
            # wrap around to beginning of array
            start -= self.full_size
            stop -= self.full_size

            if self.padding:
                # copy values from end of base array to beginning
                final_values = range(
                    self.full_size - self.padding,
                    self.full_size + self.padding,
                )
                next_previous_values = range(-self.padding, self.padding)
                b_locs = [
                    range(size) for size in self._base_shape[1 : self._base_batch_dims]
                ]
                full_array = self.full
                for source, destination in zip(final_values, next_previous_values):
                    for b_loc in itertools.product(*b_locs):
                        full_array[(destination,) + b_loc] = np.asarray(
                            full_array[(source,) + b_loc]
                        )

        # update current location with modified start/stop
        new_slice: StandardIndex = slice(start, stop, 1)
        self._current_location[0] = new_slice

    def to_list(self) -> tuple[list[np.ndarray], list[int]]:
        if self._shape is None:
            self._resolve_indexing_history()

        current_location = tuple(self._current_location)
        batch_locs, feature_locs = (
            current_location[: self._base_batch_dims],
            current_location[self._base_batch_dims :],
        )

        if any(isinstance(loc, np.ndarray) for loc in feature_locs):
            raise IndexError(
                "No advanced indexing permitted within feature dimensions."
            )

        # create a list of graphs to be concatenated into one large "batch"
        graphs: list[np.ndarray] = []
        num_elements: list[int] = []

        if any(isinstance(loc, np.ndarray) for loc in batch_locs):
            # advanced indexing: zip arrays and iterate over them together
            # TODO: also iterate over product of this zip any slices present
            batch_locs = zip(
                *(
                    iter(loc) if isinstance(loc, np.ndarray) else itertools.repeat(loc)
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
            # use neg_from_end=True here to ensure that slice doesn't get
            # extended beyond the size of the point cloud
            n_loc = compose_indices(
                n_slice, feature_locs[0], self._flattened_size, neg_from_end=False
            )
            real_loc = b_loc + (n_loc,) + feature_locs[1:]
            graph = self._base_array[real_loc]
            graphs.append(graph)
            num_elements.append(graph.shape[0])

        return graphs, num_elements

    def to_ndarray(self) -> ArrayDict[np.ndarray]:
        graphs, num_elements = self.to_list()
        array = np.concatenate(graphs) if len(graphs) > 1 else graphs[0]
        ptr = np.cumsum(num_elements, dtype=np.int64)
        ptr = np.insert(ptr, 0, 0)  # insert 0 at beginning of ptr
        assert array.shape[0] == ptr[-1]
        # TODO: hard-coded that JaggedArray is point/node positions
        # what about point/node features?
        return ArrayDict({"pos": array, "ptr": ptr})

    def __array__(self, dtype=None) -> np.ndarray:
        graphs, _ = self.to_list()
        array = np.concatenate(graphs) if len(graphs) > 1 else graphs[0]
        if dtype is not None:
            array = array.astype(dtype, copy=False)

        return array


class InheritedMemoryJaggedArray(
    InheritedMemoryArray, JaggedArray, kind="jagged", storage="inherited"
):
    pass


class SharedMemoryJaggedArray(
    SharedMemoryArray, JaggedArray, kind="jagged", storage="shared"
):
    pass


def slice_to_range(slice_: slice) -> range:
    start, stop, step = slice_.start, slice_.stop, slice_.step
    stop = -1 if stop is None else stop
    return range(start, stop, step)
