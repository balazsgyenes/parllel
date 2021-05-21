from typing import Union, Tuple, Any

import numpy as np


class Array:
    """Abstracts memory management for large arrays.
    """
    def __init__(self,
        shape: Tuple[int],
        dtype: np.dtype,
    ) -> None:
        self._shape = shape
        assert dtype is not np.object_, "Data type should not be object."
        self._dtype = dtype

    def initialize(self):
        # initialize buffer in either local or shared memory
        self._array = np.zeros(shape=self._shape, dtype=self._dtype)
        
    def __getitem__(self, location: Union[Union[int, slice, Ellipsis], Tuple[Union[int, slice, Ellipsis]]]):
        
        item = self._array[location]
        # TODO: save location for later
        return item
    
    def __setitem__(self, location: Union[Union[int, slice, Ellipsis], Tuple[Union[int, slice, Ellipsis]]], value: Any):
        self._array[location] = value

    def __array__(self, dtype = None):
        if dtype is None:
            return self._array
        else:
            return self._array.astype(dtype, copy=False)
