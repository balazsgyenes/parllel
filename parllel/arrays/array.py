from __future__ import annotations
import copy
from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray

from parllel.buffers.buffer import Buffer, Indices


class Array(Buffer):
    """Abstracts memory management for large arrays.

    Arrays must be initialized before using them.

    Example:
        >>> array = Array(shape=(4, 4, 4), dtype=np.float32)
        >>> array.initialize()
        >>> array[:, slice(1, 3), 2] = 5.

    TODO:
        - enforce dtype to be an actual np.dtype
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        self.shape = shape
        dtype = np.dtype(dtype)
        assert dtype != np.object_, "Data type should not be object."
        self.dtype = dtype

        # initialize numpy array
        self._array: NDArray = np.zeros(shape=self.shape, dtype=self.dtype)

        self._buffer_id: int = id(self)
        self._index_history: List[Indices] = []

    def __getitem__(self, location: Indices) -> Array:
        # index contained nparray to verify that location is well-formed
        nparray = self._array[location]
        # new Array object initialized through a (shallow) copy. Attributes
        # that differ between self and result are modified next. This allows
        # subclasses to override and only handle additional attributes that
        # need to be modified.
        result = type(self).__new__(type(self))
        result.__dict__.update(self.__dict__)
        # assign indexed nparray to result
        result._array = nparray
        # assign new shape
        result.shape = nparray.shape
        # assign copy of _index_history with additional element for this
        # indexing operation
        result._index_history = self._index_history + [location]
        return result

    def __setitem__(self, location: Indices, value: Any) -> None:
        self._array[location] = value

    def __array__(self, dtype=None) -> NDArray:
        if dtype is None:
            return np.atleast_1d(self._array)
        else:
            return np.atleast_1d(self._array).astype(dtype, copy=False)

    def __repr__(self) -> str:
        if hasattr(self, "_array"):
            return repr(self.__array__())
        else:
            return f"Uninitialized {type(self).__name__} object: " \
                   f"shape={self.shape}, dtype={np.dtype(self.dtype).name}."

    def __bool__(self) -> bool:
        return bool(self._array)

    def __eq__(self, o: object) -> NDArray:
        return self._array == o
