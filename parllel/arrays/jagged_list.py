from __future__ import annotations

import itertools
from typing import Literal

import numpy as np

from parllel.arrays.array import Array
from parllel.arrays.indices import (Location, StandardIndex, add_locations,
                                    init_location)
from parllel.arrays.jagged import JaggedArray


class JaggedArrayList(Array, kind="jagged_list"):
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

        self.dtype = dtype
        self.padding = padding
        self.on_overflow = on_overflow
        self.full_size = full_size

        # shape that base array appears to be
        self._base_shape = (
            (full_size + 2 * padding,)
            + batch_shape[1:]
            + (max_mean_num_elem,)
            + feature_shape
        )
        self._base_batch_dims = len(batch_shape)

        self._current_location = init_location(self._base_shape)
        # start with first array
        init_slice = slice(padding, batch_shape[0] + padding)
        self._unresolved_indices: list[Location] = [init_slice]
        self._resolve_indexing_history()

        self._rotatable = True

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
        self._current_array_iter = itertools.cycle(range(len(self.jagged_arrays)))
        self._current_array_idx = next(self._current_array_iter)

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
            array_idx, entry_idx = divmod(
                t - self.padding, self.block_size + self.padding
            )
            if array_idx != self._current_array_idx:
                raise ValueError("Setting values outside of the active JaggedArray is currently not supported.")
            new_destination = (entry_idx, *destination[1:])
            self.jagged_arrays[array_idx][new_destination] = value

        elif isinstance(destination[0], np.ndarray):
            raise NotImplementedError
            t = destination[0]
            array_idx, entry_idx = np.divmod(t - self.padding, self.block_size + self.padding)
            if np.any(array_idx != self._current_array_idx):
                raise ValueError("Setting values outside of the active JaggedArray is currently not supported.")
            new_indices = (entry_idx, *destination[1:])
            self.jagged_arrays[self._current_array_idx][new_indices] = value
            # if entry_idx == self.last + 1:
            #     self.jagged_arrays[array_idx + 1][(-1, b)] = value

    # def __getitem__(self, indices: Location):
    #     if self._shape is None:
    #         self._resolve_indexing_history()
    #     return self.jagged_arrays[self._current_array_idx][indices]

    def __array__(self, dtype=None) -> np.ndarray:
        if self._shape is None:
            self._resolve_indexing_history()
        if isinstance(self._current_location[0], int):
            array_idx, entry_idx = divmod(self._current_location[0] - self.padding, self.block_size)
            current_location = tuple([entry_idx] + self._current_location[1:])
            graphs = self.jagged_arrays[array_idx][current_location].to_list() #type: ignore

        elif isinstance(self._current_location[0], np.ndarray):
            array_idx, entry_idx = np.divmod(self._current_location[0] - self.padding, self.block_size)
            graphs = []
            current_location = [entry_idx] + self._current_location[1:]
            for i, array in enumerate(self.jagged_arrays):
                mask = array_idx == i
                new_current_location = tuple([loc[mask] if isinstance(loc, np.ndarray) else loc for loc in current_location])
                #current_location = tuple([entry_idx[array_idx == i]] + self._current_location[1:])
                graphs.extend(array[new_current_location].to_list())

            #graphs = [array.to_list() for array in self.jagged_arrays]

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
        next_current_array = next(self._current_array_iter)

        if start >= self.full_size:
            # wrap around to beginning of array
            start -= self.full_size
            stop -= self.full_size

            # if self.padding:
            #     # copy values from end of base array to beginning
            #     final_values = range(
            #         self.full_size - self.padding,
            #         self.full_size + self.padding,
            #     )
            #     next_previous_values = range(-self.padding, self.padding)
            #     b_locs = [
            #         range(size) for size in self._base_shape[1 : self._base_batch_dims]
            #     ]
            #     full_array = self.full

            #     for source, destination in zip(final_values, next_previous_values):
            #         for b_loc in itertools.product(*b_locs):
            #             full_array[(destination,) + b_loc] = np.asarray(
            #                 full_array[(source,) + b_loc]
            #             )
            #     self._current_array = 0
        if self.padding:
            final_values = range(
                self.block_size - self.padding, self.block_size + self.padding
            )
            next_previous_values = range(-self.padding, self.padding)
            b_locs = [
                range(size) for size in self._base_shape[1 : self._base_batch_dims]
            ]
            for source, destination in zip(final_values, next_previous_values):
                for b_loc in itertools.product(*b_locs):
                    self.jagged_arrays[next_current_array][
                        (destination,) + b_loc
                    ] = np.asarray(
                        self.jagged_arrays[self._current_array_idx][(source,) + b_loc]
                    )
        self._current_array_idx = next_current_array
        new_slice: StandardIndex = slice(start, stop, 1)
        self._current_location[0] = new_slice
