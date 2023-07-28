from __future__ import annotations

import itertools
from typing import Literal

import numpy as np

from parllel.arrays.array import Array
from parllel.arrays.indices import Location, StandardIndex, init_location
from parllel.arrays.jagged import JaggedArray


class JaggedArrayList(Array, kind="jagged_list"):
    kind = "jagged_list"

    def __init__(
        self,
        feature_shape: tuple[int, ...],
        dtype: np.dtype,
        *,
        batch_shape: tuple[int, ...],
        kind: str | None = None,
        storage: str | None = None,
        padding: int = 0,
        full_size: int,
        on_overflow: Literal["drop", "resize", "wrap"] = "drop",
    ):
        if full_size % batch_shape[0] != 0:
            raise ValueError(
                f"Full size ({full_size}) not divisible by size_T ({batch_shape[0]})"
            )

        num_arrays = full_size // batch_shape[0]
        dtype = np.dtype(dtype)

        shape = batch_shape + feature_shape

        if dtype == np.object_:
            raise ValueError("Data type should not be object.")

        self.dtype = dtype
        self.padding = padding
        self.on_overflow = on_overflow
        self.full_size = full_size

        # shape that base array appears to be
        self._base_shape = (full_size + 2 * padding,) + shape[1:]
        self._base_batch_dims = len(batch_shape)

        self._current_location = init_location(self._base_shape)
        init_slice = slice(padding, shape[0] + padding)
        self._unresolved_indices: list[Location] = [init_slice]
        self._resolve_indexing_history()

        # TODO: why do we need this?
        self._rotatable = True

        # self._flattened_size = self._base_shape[0] * feature_shape[0]

        self.jagged_arrays = [
            JaggedArray(
                feature_shape=feature_shape,
                dtype=dtype,
                batch_shape=batch_shape,
                kind="jagged",
                storage=storage,
                padding=padding,
                full_size=None,
                on_overflow=on_overflow,
            )
            for _ in range(num_arrays)
        ]

    def __setitem__(
        self, indices: Location, value: float | int | np.number | list | JaggedArray
    ) -> None:
        if self._shape is None:
            self._resolve_indexing_history()

        if indices is ...:
            for array in self.jagged_arrays:
                array[...] = value
        elif isinstance(indices, int):
            array_idx, entry_idx = divmod(indices, len(self.jagged_arrays))
            self.jagged_arrays[array_idx][entry_idx] = value
        else:  # indices is a tuple
            # TODO: remove asserts
            assert isinstance(indices, tuple) and len(indices) == 2
            t, b = indices
            array_idx, entry_idx = divmod(t, len(self.jagged_arrays))
            new_indices = (entry_idx, b)
            self.jagged_arrays[array_idx][new_indices] = value
            pass

    def __getitem__(self, indices: Location):
        if indices is ...:
            raise NotImplementedError
        elif isinstance(indices, int):
            array_idx, entry_idx = divmod(indices, len(self.jagged_arrays))
            return self.jagged_arrays[array_idx][entry_idx]
        else:  # indices is a tuple
            # TODO: remove asserts
            assert isinstance(indices, tuple) and len(indices) == 2
            t, b = indices
            array_idx, entry_idx = divmod(t, len(self.jagged_arrays))
            new_indices = (entry_idx, b)
            return self.jagged_arrays[array_idx][new_indices]

    def __array__(self, dtype=None) -> np.ndarray:
        graphs = [array.__array__() for array in self.jagged_arrays]
        array = np.concatenate(graphs) if len(graphs) > 1 else graphs[0]
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def rotate(self) -> None:
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
                        full_array[(destination,) + b_locs] = np.asarray(
                            full_array[(source,) + b_loc]
                        )
        # update current location with modified start/stop
        new_slice: StandardIndex = slice(start, stop, 1)
        self._current_location[0] = new_slice
