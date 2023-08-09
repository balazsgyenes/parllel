# fmt: off
from __future__ import annotations

import itertools
from typing import Literal

import numpy as np

from parllel.arrays.indices import (Location, StandardIndex, add_locations,
                                    init_location)
from parllel.arrays.jagged import JaggedArray


# fmt: on
class CurrentIndex:  # we can't add attributes to objects created via object()
    pass


class JaggedArrayList(JaggedArray):  # do not register subclass
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

        if padding < 0:
            raise ValueError("Padding must be non-negative.")
        if padding > batch_shape[0]:
            raise ValueError(
                f"Padding ({padding}) cannot be greater than leading dimension {batch_shape[0]}."
            )

        assert full_size is not None and full_size > batch_shape[0]
        if full_size % batch_shape[0] != 0:
            raise ValueError(
                f"Full size ({full_size}) must be evenly divided by leading dimension {batch_shape[0]}."
            )

        self.block_size = batch_shape[0]

        self.padding = padding
        self.full_size = full_size
        num_arrays = full_size // batch_shape[0]

        self.jagged_arrays: list[JaggedArray] = [
            JaggedArray(
                batch_shape=batch_shape,
                dtype=dtype,
                max_mean_num_elem=max_mean_num_elem,
                feature_shape=feature_shape,
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
        self._current_array = CurrentIndex()
        self._current_array_idx = next(self._current_array_iter)
        self._resolve_indexing_history()

    @property
    def dtype(self) -> np.dtype:
        return self.jagged_arrays[0].dtype

    @property
    def max_mean_num_elem(self) -> int:
        return self.jagged_arrays[0].max_mean_num_elem

    @property
    def storage(self) -> str:
        return self.jagged_arrays[0].storage

    @property
    def on_overflow(self) -> str:
        return self.jagged_arrays[0].on_overflow

    @property
    def _current_array_idx(self) -> int:
        return self._current_array.index

    @_current_array_idx.setter
    def _current_array_idx(self, value: int):
        self._current_array.index = value

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
            new_destination = (entry_idx,) + destination[1:]
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

    def to_list(self) -> tuple[list[np.ndarray], list[int]]:
        if self._shape is None:
            self._resolve_indexing_history()

        if isinstance(self._current_location[0], int):
            array_idx, entry_idx = divmod(
                self._current_location[0] - self.padding, self.block_size
            )

            if array_idx == self._current_array_idx + 1 and entry_idx < self.padding:
                array_idx = self._current_array_idx
                entry_idx = self.block_size + entry_idx

            if array_idx >= len(self.jagged_arrays):
                array_idx = len(self.jagged_arrays) - 1
                entry_idx = self.block_size + entry_idx

            current_location = tuple([entry_idx] + self._current_location[1:])
            return self.jagged_arrays[array_idx][current_location].to_list()  # type: ignore

        elif isinstance(self._current_location[0], np.ndarray):
            array_idx, entry_idx = np.divmod(
                self._current_location[0] - self.padding, self.block_size
            )
            if np.any(
                mask := (
                    (array_idx == self._current_array_idx + 1)
                    & (entry_idx < self.padding)
                )
            ):
                array_idx[mask] = self._current_array_idx
                entry_idx[mask] = self.block_size + entry_idx[mask]

            if np.any(mask := (array_idx >= len(self.jagged_arrays))):
                array_idx[mask] = len(self.jagged_arrays) - 1
                entry_idx[mask] = self.block_size + entry_idx[mask]

            graphs = []
            num_elements = []
            current_location = [entry_idx] + self._current_location[1:]
            for i, array in enumerate(self.jagged_arrays):
                mask = array_idx == i
                new_current_location = tuple(
                    loc[mask] if isinstance(loc, np.ndarray) else loc
                    for loc in current_location
                )
                new_graphs, new_nums = array[new_current_location].to_list()
                graphs.extend(new_graphs)
                num_elements.extend(new_nums)

            return graphs, num_elements
        else:
            raise NotImplementedError

    def rotate(self) -> None:
        if not self._rotatable:
            raise ValueError("rotate() is only allowed on unindexed array.")

        leading_loc: slice = self._current_location[0]
        start = leading_loc.start + self.shape[0]
        stop = leading_loc.stop + self.shape[0]
        next_array_idx = next(self._current_array_iter)

        if start >= self.full_size:
            # wrap around to beginning of array
            start -= self.full_size
            stop -= self.full_size

        # update current location with modified start/stop
        new_slice: StandardIndex = slice(start, stop, 1)
        self._current_location[0] = new_slice

        if self.padding:
            final_values = range(
                self.shape[0] - self.padding,
                self.shape[0] + self.padding,
            )
            next_previous_values = range(-self.padding, self.padding)
            b_locs = [
                range(size) for size in self._base_shape[1 : self._base_batch_dims]
            ]
            for source, destination in zip(final_values, next_previous_values):
                for b_loc in itertools.product(*b_locs):
                    self.jagged_arrays[next_array_idx][
                        (destination,) + b_loc
                    ] = np.asarray(
                        self.jagged_arrays[self._current_array_idx][(source,) + b_loc]
                    )

        self._current_array_idx = next_array_idx

    def close(self):
        for array in self.jagged_arrays:
            array.close()
