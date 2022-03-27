from __future__ import annotations
from functools import reduce
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

        self._allocate()

        # the result of calling np.asarray() on the array at any time
        self._current_array: NDArray = self._base_array

        # used to enable indexing into a single element like element[:] = 0
        # set to the previous value of current_array, or the base_array
        self._previous_array: NDArray = self._base_array

        self._current_indices = None

    def _allocate(self) -> None:
        # initialize numpy array
        self._base_array: NDArray = np.zeros(shape=self._base_shape, dtype=self.dtype)

    def _resolve_indexing_history(self) -> None:
        array = self._base_array
        
        # if index history has 0 or 1 elements, this has no effect
        array = reduce(lambda arr, index: arr[index], self._index_history[:-1], array)
        self._previous_array = array
        
        # index last item in history only if there is a history
        if self._index_history:
            array = array[self._index_history[-1]]

        self._current_array = array
        self._apparent_shape = array.shape

    @property
    def shape(self):
        if self._apparent_shape is None:
            self._resolve_indexing_history()

        return self._apparent_shape

    def __getitem__(self, location: Indices) -> Array:
        # new Array object initialized through a (shallow) copy. Attributes
        # that differ between self and result are modified next. This allows
        # subclasses to override and only handle additional attributes that
        # need to be modified.
        result: Array = self.__new__(type(self))
        result.__dict__.update(self.__dict__)
        # assign *copy* of _index_history with additional element for this
        # indexing operation
        result._index_history = result._index_history + [location]
        # current array and shape are not computed until needed
        result._current_array = None
        result._apparent_shape = None
        # if self._current_array is not None, saves extra computation
        # if it is None, then it must be recomputed anyway
        result._previous_array = self._current_array
        return result

    def __setitem__(self, location: Indices, value: Any) -> None:
        if self._current_array is None:
            self._resolve_indexing_history()

        if self._apparent_shape == ():
            # Need to avoid item assignment on a scalar (0-D) array, so we assign
            # into previous array using the last indices used
            if not (location == slice(None) or location == ...):
                raise IndexError("Cannot take slice of 0-D array.")
            # in this case, there must be an index history
            location = self._index_history[-1]
            destination = self._previous_array
        else:
            destination = self._current_array

        destination[location] = value

    def __array__(self, dtype=None) -> NDArray:
        if self._current_array is None:
            self._resolve_indexing_history()
        
        array = np.asarray(self._current_array)  # promote scalars to 0d arrays
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    @property
    def current_indices(self):
        if self._current_indices == None:
            self._current_indices = compute_current_indices(
                self._base_array, self._current_array,
            )
        return self._current_indices

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


def compute_current_indices(base_array, current_array):
    # make strides positive 
    # current_strides = tuple(abs(stride) for stride in self._current_array.strides)
    base_shape = base_array.shape
    base_strides = base_array.strides
    curr_shape = current_array.shape
    curr_strides = current_array.strides
    current_indices = [None for _ in base_shape]

    current_pointer = current_array.__array_interface__["data"][0]
    base_pointer = base_array.__array_interface__["data"][0]
    offset = current_pointer - base_pointer
    dim_offsets = [0 for _ in base_shape]
    for dim, base_stride in enumerate(base_strides):
        dim_offsets[dim] = offset // base_stride
        offset %= base_stride
    assert offset == 0

    for dim, (base_size, base_stride) in enumerate(zip(base_shape, base_strides)):
        # search for the corresponding stride in current_array
        # base_stride_(dim-1) > abs(current_stride_(dim)) > base_stride_(dim), if it exists
        # the absolute value is required because the current stride might be negative
        for i, (curr_size, curr_stride) in enumerate(zip(curr_shape, curr_strides)):
            # if dim==0, skip check against previous base_stride
            if abs(curr_stride) >= base_stride and (
                dim == 0 or base_strides[dim-1] > curr_stride):
                # stride exists, so index is at least not an integer
                # stride can only have been changed by slicing with a step != 1
                step = curr_stride // base_stride
                start = dim_offsets[dim]
                stop = start + curr_size * step
                # if step > 0:
                #     start = start if start != 0 else None
                #     stop = stop if stop != base_size else None
                # else:
                #     start -= base_size
                #     start = start if start != -1 else None
                #     stop = stop if stop != -1 else None
                # step = step if step != 1 else None
                current_indices[dim] = slice(start, stop, step)
                break
        else:
            # stride does not exist, so index must be integer
            # calculate index using offset
            current_indices[dim] = dim_offsets[dim]

    return current_indices


if __name__ == "__main__":
    array = np.zeros(shape=(8,4,3,96,96))
    subarray = array[::2, ::-3, 2, :48, 48:]

    indices = compute_current_indices(array, subarray)
    print(indices)