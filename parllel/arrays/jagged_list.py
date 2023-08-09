# fmt: off
from __future__ import annotations

import itertools
from typing import Literal, TypeVar

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

        t_index = destination[0]
        if isinstance(t_index, slice):
            raise NotImplementedError

        array_idx, entry_idx = divmod_with_padding(
            index=t_index - self.padding,
            block_size=self.block_size,
            active_block=self._current_array_idx,
            n_blocks=len(self.jagged_arrays),
            padding=self.padding,
        )
        new_destination = (entry_idx,) + destination[1:]
        if isinstance(t_index, int):
            self.jagged_arrays[array_idx][new_destination] = value

        elif isinstance(t_index, np.ndarray):
            for i, array in enumerate(self.jagged_arrays):
                mask = array_idx == i
                new_current_location = tuple(
                    loc[mask] if isinstance(loc, np.ndarray) else loc
                    for loc in new_destination
                )
                new_value = value[mask] if isinstance(value, np.ndarray) else value
                array[new_current_location] = new_value
        else:
            raise ValueError(f"Unknown index type {type(t_index).__name__}")

    def to_list(self) -> tuple[list[np.ndarray], list[int]]:
        if self._shape is None:
            self._resolve_indexing_history()

        t_index = self._current_location[0]
        if isinstance(t_index, slice):
            raise NotImplementedError

        array_idx, entry_idx = divmod_with_padding(
            index=t_index - self.padding,
            block_size=self.block_size,
            active_block=self._current_array_idx,
            n_blocks=len(self.jagged_arrays),
            padding=self.padding,
        )
        current_location = (entry_idx,) + tuple(self._current_location[1:])

        if isinstance(t_index, int):
            return self.jagged_arrays[array_idx][current_location].to_list()  # type: ignore

        elif isinstance(t_index, np.ndarray):
            graphs = []
            num_elements = []
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
            raise ValueError(f"Unknown index type {type(t_index).__name__}")

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

    def __repr__(self) -> str:
        try:
            return super().__repr__()
        except NotImplementedError:
            return (
                type(self).__name__ + "("
                "..."
                f", storage={self.storage}"
                f", dtype={self.dtype.name}"
                f", padding={self.padding}"
                ")"
            )


IndexType = TypeVar("IndexType", int, slice, np.ndarray)


def divmod_with_padding(
    index: IndexType,
    block_size: int,
    active_block: int,
    n_blocks: int,
    padding: int,
) -> tuple[IndexType, IndexType]:
    if isinstance(index, int):
        if (
            active_block * block_size - padding
            <= index
            < (active_block + 1) * block_size + padding
        ):
            # index is contained within active block +/- padding
            return active_block, index - active_block * block_size
        elif index < 0:
            # index is in leading padding of first block
            return 0, index
        elif index > n_blocks * block_size:
            # index is in trailing padding of final block
            return n_blocks - 1, index - (n_blocks - 1) * block_size
        else:
            # index is not within any of the accessible padding regions
            return divmod(index, block_size)

    elif isinstance(index, np.ndarray):
        # create new arrays major and minor so we don't modify index
        # divmod is actually equivalent to np.divmod
        major, minor = np.divmod(index, block_size)

        # index is contained within active block +/- padding
        mask = (active_block * block_size - padding <= index) & (
            index < (active_block + 1) * block_size + padding
        )
        major[mask], minor[mask] = active_block, index[mask] - active_block * block_size

        # index is in leading padding of first block
        mask = index < 0
        major[mask], minor[mask] = 0, index[mask]

        # index is in trailing padding of final block
        mask = index > n_blocks * block_size
        major[mask], minor[mask] = (
            n_blocks - 1,
            index[mask] - (n_blocks - 1) * block_size,
        )

        return major, minor

    elif isinstance(index, slice):
        raise NotImplementedError

    else:
        raise ValueError(f"Unknown index type {type(t_index).__name__}")
