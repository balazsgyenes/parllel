from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np

from parllel import Array, ArrayDict
from parllel.arrays.indices import (Location, StandardIndex, add_locations,
                                    batch_dims_from_location, init_location)
from parllel.arrays.jagged import JaggedArray


class JaggedArrayList(JaggedArray, kind="jagged_list"):
    kind = "jagged_list"

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

        self.block_size = batch_shape[0]
        self.max_mean_num_elem = max_mean_num_elem

        self.dtype = dtype
        self.padding = padding
        self.on_overflow = on_overflow
        self.full_size = full_size
        num_arrays = full_size // batch_shape[0]

        self.jagged_arrays = [
            JaggedArray(
                batch_shape=batch_shape,
                dtype=dtype,
                max_mean_num_elem=max_mean_num_elem,
                feature_shape=feature_shape,
                kind="jagged",
                storage=storage,
                padding=padding,
                full_size=None,
                on_overflow=on_overflow,
            )
            for _ in range(num_arrays)
        ]
        # shape that base array appears to be
        self._base_shape = (
            (full_size + (2 * padding),)
            + batch_shape[1:]
            + (max_mean_num_elem,)
            + feature_shape
        )
        self._base_batch_dims = len(batch_shape)

        self._current_location = init_location(self._base_shape)
        # start with first array
        init_slice = slice(padding, batch_shape[0] + padding)
        self._unresolved_indices: list[Location] = [init_slice]

        self._rotatable = True

        self._current_array_iter = itertools.cycle(range(len(self.jagged_arrays)))
        self._current_array_idx = next(self._current_array_iter)
        self._resolve_indexing_history()
        self.access_padding = False

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

        return super().new_array(*args, **kwargs)

    def reset(self):
        super().reset()
        # the current jagged array is the last one in the list
        self._current_array_idx = len(self.jagged_arrays) - 1
        # next time next() is called, the iterator is at 0
        self._current_array_iter = itertools.cycle(range(len(self.jagged_arrays)))

    def __setitem__(
        self, indices: Location, value: float | int | np.number | list | JaggedArray
    ) -> None:
        if self._shape is None:
            self._resolve_indexing_history()

        destination = tuple(
            add_locations(
                self._current_location,
                indices,
                self._base_shape,
                neg_from_end=False,
            )
        )

        if isinstance(destination[0], int):
            t = destination[0]
            array_idx, entry_idx = divmod(t - self.padding, self.block_size)

            if array_idx == self._current_array_idx + 1 and entry_idx < self.padding:
                array_idx = self._current_array_idx
                entry_idx = self.block_size + entry_idx

            if array_idx != self._current_array_idx:
                raise ValueError(
                    "Setting values outside of the active JaggedArray is currently not supported."
                )
            new_destination = (entry_idx, *destination[1:])
            self.jagged_arrays[array_idx][new_destination] = value

        elif isinstance(destination[0], np.ndarray):
            raise NotImplementedError
            t = destination[0]
            array_idx, entry_idx = np.divmod(
                t - self.padding, self.block_size + self.padding
            )
            if np.any(array_idx != self._current_array_idx):
                raise ValueError(
                    "Setting values outside of the active JaggedArray is currently not supported."
                )
            new_indices = (entry_idx, *destination[1:])
            self.jagged_arrays[self._current_array_idx][new_indices] = value
            # if entry_idx == self.last + 1:
            #     self.jagged_arrays[array_idx + 1][(-1, b)] = value
        else:
            raise NotImplementedError

    # def __getitem__(self, indices: Location):
    #     if self._shape is None:
    #         self._resolve_indexing_history()
    #     return self.jagged_arrays[self._current_array_idx][indices]

    def __array__(self, dtype=None) -> np.ndarray:
        if self._shape is None:
            self._resolve_indexing_history()
        if isinstance(self._current_location[0], int):
            array_idx, entry_idx = divmod(
                self._current_location[0] - self.padding, self.block_size
            )
            if array_idx >= len(self.jagged_arrays):
                array_idx = len(self.jagged_arrays) - 1
                entry_idx = self.block_size - entry_idx

            current_location = tuple([entry_idx] + self._current_location[1:])
            graphs, current_ptrs = self.jagged_arrays[array_idx][current_location].to_list()  # type: ignore
            current_ptrs = np.array(current_ptrs)

        elif isinstance(self._current_location[0], np.ndarray):
            array_idx, entry_idx = np.divmod(
                self._current_location[0] - self.padding, self.block_size
            )
            if np.any(mask := (array_idx >= len(self.jagged_arrays))):
                array_idx[mask] = len(self.jagged_arrays) - 1
                entry_idx[mask] = self.block_size - entry_idx[mask]

            graphs = []
            current_ptrs = []
            repeat = []
            offset = []
            num_nodeses = []
            current_location = [entry_idx] + self._current_location[1:]
            for i, array in enumerate(self.jagged_arrays):
                mask = array_idx == i
                new_current_location = tuple(
                    [
                        loc[mask] if isinstance(loc, np.ndarray) else loc
                        for loc in current_location
                    ]
                )
                # current_location = tuple([entry_idx[array_idx == i]] + self._current_location[1:])
                graph, ptr = array[new_current_location].to_list()
                if len(ptr) > 1:
                    current_ptrs.append(ptr[1:])  # we don't need the first pointer to 0
                    repeat.append(len(ptr) - 1)
                    offset.append(ptr[-1])
                    num_nodes = ptr[1:] - ptr[:-1]
                    num_nodeses.append(num_nodes)
                    graphs.extend(graph)

            num_nodeses = np.concatenate(num_nodeses)
            num_nodeses = np.cumsum(num_nodeses)
            num_nodeses = np.insert(num_nodeses, 0, 0)
        else:
            raise NotImplementedError

        self._current_ptrs = current_ptrs.flatten()
        array = np.concatenate(graphs) if len(graphs) > 1 else graphs[0]
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def rotate(self) -> None:
        if not self._rotatable:
            raise ValueError("rotate() is only allowed on unindexed array.")

        leading_loc: slice = self._current_location[0]
        start = leading_loc.start + self.block_size
        stop = leading_loc.stop + self.block_size
        next_array_idx = next(self._current_array_iter)

        if start >= self.full_size:
            # wrap around to beginning of array
            start -= self.full_size
            stop -= self.full_size

        if self.padding:
            final_values = range(
                self.block_size - self.padding, self.block_size + self.padding
            )
            next_previous_values = range(-self.padding, self.padding)
            b_locs = [
                range(size) for size in self._base_shape[1 : self._base_batch_dims]
            ]
            self.access_padding = True
            for source, destination in zip(final_values, next_previous_values):
                for b_loc in itertools.product(*b_locs):
                    self.jagged_arrays[next_array_idx][
                        (destination,) + b_loc
                    ] = np.asarray(
                        self.jagged_arrays[self._current_array_idx][(source,) + b_loc]
                    )
        self.access_padding = False
        self._current_array_idx = next_array_idx
        new_slice: StandardIndex = slice(start, stop, 1)
        self._current_location[0] = new_slice

    def to_ndarray(self) -> ArrayDict[np.ndarray]:
        # TODO: hard-coded that JaggedArray is point/node positions
        # what about point/node features?
        data = self.__array__()
        return ArrayDict(
            {
                "pos": data,
                "ptr": self._current_ptrs,  # updated during execution of __array__
            }
        )
