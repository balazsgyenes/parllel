from typing import Any, Tuple, Union

import numpy as np
from nptyping import NDArray


Index = Union[int, slice, type(Ellipsis)]
Indices = Union[Index, Tuple[Index, ...]]


class Array:
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
        self._shape = shape
        dtype = np.dtype(dtype)
        assert dtype is not np.object_, "Data type should not be object."
        self._dtype = dtype

    def initialize(self) -> None:
        # initialize numpy array
        self._array = np.zeros(shape=self._shape, dtype=self._dtype)

    def __getitem__(self, location: Indices) -> NDArray:
        return self._array[location]

    def __setitem__(self, location: Indices, value: Any) -> None:
        self._array[location] = value

    def __array__(self, dtype=None) -> NDArray:
        if dtype is None:
            return self._array
        else:
            return self._array.astype(dtype, copy=False)

    def __repr__(self) -> str:
        if hasattr(self, "_array"):
            return repr(self._array)
        else:
            return f"Uninitialized {type(self).__name__} object: " f"shape={self._shape}, dtype={np.dtype(self._dtype).name}."
