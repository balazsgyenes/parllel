from __future__ import annotations
from itertools import chain
from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray

from parllel.buffers.buffer import Buffer, Indices
from .indices import compute_indices, slicify_final_index, does_index_scalar


class Array(Buffer):
    """An object wrapping a numpy array for use in sampling. An Array remembers
    indexing operations used to get subarrays. Math operations are generally
    not supported, use `np.asarray(arr)` to get the underlying numpy array.

    Example:
        >>> array = Array(shape=(4, 4, 4), dtype=np.float32)
        >>> array[:, slice(1, 3), 2] = 5.
    """

    def __init__(self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        if not shape:
            raise ValueError("Non-empty shape required.")
        self._base_shape = shape
        self._apparent_shape = shape

        dtype = np.dtype(dtype)
        if dtype == np.object_:
            raise TypeError("Data type should not be object.")
        self.dtype = dtype

        self._buffer_id: int = id(self)
        self._index_history: List[Indices] = []
        self._unresolved_indices: List[Indices] = []

        self._allocate()

        # the result of calling np.asarray() on the array at any time
        self._current_array: NDArray = self._base_array
        self._current_indices = [slice(None) for _ in self._base_shape]

    def _allocate(self) -> None:
        # initialize numpy array
        self._base_array: NDArray = np.zeros(shape=self._base_shape, dtype=self.dtype)

    def _resolve_indexing_history(self) -> None:
        for location in self._unresolved_indices:
            if does_index_scalar(self._apparent_shape, location):
                # sneakily turn final index into a slice so that indexing a
                # scalar (which results in a copy) is avoided
                location = slicify_final_index(location)                
                self._current_array = self._current_array[location]
                self._apparent_shape = ()

            elif not self._apparent_shape and not location == ...:
                raise IndexError("invalid index to scalar variable.")

            else:
                self._current_array = self._current_array[location]
                self._apparent_shape = self._current_array.shape

        self._index_history += self._unresolved_indices
        self._unresolved_indices = []

    @property
    def shape(self):
        self._resolve_indexing_history()
        return self._apparent_shape

    @property
    def index_history(self) -> Tuple[Indices, ...]:
        return tuple(chain(self._index_history, self._unresolved_indices))

    @property
    def current_indices(self):
        if self._current_indices is None:

            self._resolve_indexing_history()

            self._current_indices = compute_indices(
                self._base_array, self._current_array)

        return self._current_indices

    def __getitem__(self, location: Indices) -> Array:
        # new Array object initialized through a (shallow) copy. Attributes
        # that differ between self and result are modified next. This allows
        # subclasses to override and only handle additional attributes that
        # need to be modified.
        result: Array = self.__new__(type(self))
        result.__dict__.update(self.__dict__)
        # assign *copy* of index_history
        result._index_history = result._index_history.copy()
        # assign *copy* of _unresolved_indices with additional element for this
        # indexing operation
        result._unresolved_indices = result._unresolved_indices + [location]
        # current array and shape are not computed until needed
        result._current_indices = None
        return result

    def __setitem__(self, location: Indices, value: Any) -> None:
        self._resolve_indexing_history()

        if not self._apparent_shape and not location == ...:
            raise IndexError("invalid index to scalar variable.")

        self._current_array[location] = value

    def __array__(self, dtype=None) -> NDArray:
        self._resolve_indexing_history()

        array = self._current_array

        if self._apparent_shape == ():
            # compensate for final index replaced with slice
            array = array[0]

        # this function must return an array, so promote scalars to 0d arrays
        array = np.asarray(array)
        
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def __repr__(self) -> str:
        return repr(self.__array__())

    def __bool__(self) -> bool:
        return bool(self.__array__())

    def __eq__(self, o: object) -> NDArray:
        return self.__array__() == o

    def close(self):
        pass

    def destroy(self):
        pass
