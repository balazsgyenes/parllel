from __future__ import annotations
from functools import partial, reduce
from itertools import chain
from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray

from parllel.buffers.buffer import Buffer, Indices
from .indices import add_indices


class Array(Buffer):
    """An object wrapping a numpy array for use in sampling. An Array remembers
    indexing operations used to get subarrays. Math operations are generally
    not supported, use `np.asarray(arr)` to get the underlying numpy array.

    Example:
        >>> array = Array(shape=(4, 4, 4), dtype=np.float32)
        >>> array[:, slice(1, 3), 2] = 5.

    TODO:
        - enforce dtype to be an actual np.dtype
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
        self._unresolved_index_history: List[Indices] = []

        self._allocate()

        self._current_indices = [slice(None) for _ in self._base_shape]
        self._add_indices = partial(add_indices, self._base_shape)

    def _allocate(self) -> None:
        # initialize numpy array
        self._base_array: NDArray = np.zeros(shape=self._base_shape, dtype=self.dtype)

    def _resolve_indexing_history(self) -> None:
        self._current_indices = reduce(self._add_indices, self._unresolved_index_history,
            list(self._current_indices))
        self._apparent_shape = shape_from_indices(self._base_shape, self._current_indices)
        self._index_history += self._unresolved_index_history
        self._unresolved_index_history = []

    @property
    def index_history(self) -> Tuple[Indices, ...]:
        return tuple(chain(self._index_history, self._unresolved_index_history))

    @property
    def shape(self):
        if self._unresolved_index_history:
            self._resolve_indexing_history()
        return self._apparent_shape

    @property
    def current_indices(self):
        return tuple(self._current_indices)

    def __getitem__(self, location: Indices) -> Array:
        # new Array object initialized through a (shallow) copy. Attributes
        # that differ between self and result are modified next. This allows
        # subclasses to override and only handle additional attributes that
        # need to be modified.
        result: Array = self.__new__(type(self))
        result.__dict__.update(self.__dict__)
        # assign *copy* of _index_history with additional element for this
        # indexing operation
        result._unresolved_index_history = result._unresolved_index_history + [location]
        return result

    def __setitem__(self, location: Indices, value: Any) -> None:
        if self._unresolved_index_history:
            self._resolve_indexing_history()

        # TODO: copy required because current_indices modified by function add_indices
        location = tuple(add_indices(self._base_shape, list(self._current_indices), location))

        self._base_array[location] = value

    def __array__(self, dtype=None) -> NDArray:
        if self._unresolved_index_history:
            self._resolve_indexing_history()
        
        current_indices = tuple(self._current_indices)
        array = np.asarray(self._base_array[current_indices])  # promote scalars to 0d arrays
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
