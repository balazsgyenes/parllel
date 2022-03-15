from __future__ import annotations
import copy
from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray

from parllel.buffers.buffer import Buffer, Indices


class Array(Buffer):
    """An object wrapping a numpy array for use in sampling. An Array remembers
    indexing operations used to get subarrays. Math operations are generally
    not supported, use `np.asarray(arr)` to get the underlying numpy array.

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
        assert shape, "Non-empty shape required."
        self._base_shape = shape
        self._apparent_shape = shape

        dtype = np.dtype(dtype)
        assert dtype != np.object_, "Data type should not be object."
        self.dtype = dtype

        self._buffer_id: int = id(self)
        self._index_history: List[Indices] = []
        self._allocate()

    def _allocate(self) -> None:
        # initialize numpy array
        self._array: NDArray = np.zeros(shape=self._base_shape, dtype=self.dtype)

    @property
    def shape(self):
        return self._apparent_shape

    def __getitem__(self, location: Indices) -> Array:
        array: NDArray = self._array
        if self._index_history:
            # apply last index before indexing again
            array: NDArray = array[self._index_history[-1]]
        
        # index contained nparray to verify that location is well-formed
        subarray: NDArray = array[location]
        # new Array object initialized through a (shallow) copy. Attributes
        # that differ between self and result are modified next. This allows
        # subclasses to override and only handle additional attributes that
        # need to be modified.
        result: Array = self.__new__(type(self))
        result.__dict__.update(self.__dict__)
        # assign array prior to most recent indexing
        result._array = array
        # assign apparent shape of indexed subarray
        result._apparent_shape = subarray.shape
        # assign *copy* of _index_history with additional element for this
        # indexing operation
        result._index_history = result._index_history + [location]
        return result

    def __setitem__(self, location: Indices, value: Any) -> None:
        # Need to avoid item assignment on a scalar (0-D) array, so we defer
        # the most recent indexing operation until we need it
        current_array = self._array
        if self._apparent_shape == ():
            assert location == slice(None), "Cannot take slice of 0-D array."
            location = self._index_history[-1] # in this case, there must be an index history
        elif self._index_history:
            # apply last index before item assignment
            current_array = current_array[self._index_history[-1]]

        current_array[location] = value

    def __array__(self, dtype=None) -> NDArray:
        array = self._array
        if self._index_history:
            array = array[self._index_history[-1]]
        array = np.asarray(array)  # promote scalars to 0d arrays
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


def concatenate_indices(full_shape, current_indices: List, new_indices: Tuple):
    if not isinstance(new_indices, tuple):
        new_indices = (new_indices,)

    i = 0
    j = 0
    while j < len(new_indices):
        if i >= ndim:
            raise IndexError
        if i < len(current_indices):
            current_index = current_indices[i]
        else:
            # TODO: should the default index be None or slice(None)?
            current_index = None
            current_indices.append(current_index)

        if isinstance(current_index, int):
            i += 1
            continue

        new_index = new_indices[j]

        if current_index == None:
            current_indices[i] = new_index
            i += 1
            j += 1
            continue

        if isinstance(new_index, int):
            current_indices[i] = slice_plus_int(current_index, new_index)
            i += 1
            j += 1
            continue

        current_indices[i] = slice_plus_slice(current_index, new_index)


def slice_plus_int(s: slice, n: int):
    s = slice(*s.indices(1))
    return s.start + n * s.step

def slice_plus_slice(s1: slice, s2: slice, list_length: int):
    s1 = slice(*s1.indices(1))
    s2 = slice(*s2.indices(list_length))
    return slice(
        s1.start + s2.start * s1.step,  # start
        s1.start + s2.stop * s1.step,   # stop
        s1.step * s2.step,              # step
    )