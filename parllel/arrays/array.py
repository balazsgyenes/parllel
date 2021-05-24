from typing import Any, Tuple, Union

import numpy as np
from nptyping import NDArray


Index = Union[int, slice, type(Ellipsis)]
Indices = Union[Index, Tuple[Index, ...]]


class Array:
    """Abstracts memory management for large arrays.
    """
    def __init__(self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        self._shape = shape
        assert dtype is not np.object_, "Data type should not be object."
        self._dtype = dtype

    def initialize(self) -> None:
        # initialize buffer in either local or shared memory
        self._array = np.zeros(shape=self._shape, dtype=self._dtype)
        
    def __getitem__(self, location: Indices) -> NDArray:
        return self._array[location]
    
    def __setitem__(self, location: Indices, value: Any) -> None:
        self._array[location] = value

    def __array__(self, dtype = None) -> NDArray:
        if dtype is None:
            return self._array
        else:
            return self._array.astype(dtype, copy=False)
