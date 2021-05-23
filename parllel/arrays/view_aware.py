# required for using ViewAwareArray for type annotations within ViewAwareArray
from __future__ import annotations
from copy import deepcopy
from typing import Union, List, Tuple, Any

Index = Union[int, slice, type(Ellipsis)]

import numba
import numpy as np
from nptyping import NDArray

class ViewAwareArray(np.ndarray):
    def __new__(cls, input_array: NDArray):
        obj = np.asarray(input_array).view(cls)
        obj._view_indices = []
        return obj

    def __array_finalize__(self, obj):
        # called via explicit constructor
        if obj is None: return

        # called by view casting or new-from-template
        if self.base is not None:
            self._view_indices: List[Union[None, slice, int]] = deepcopy(getattr(obj, "_view_indices", []))
        else:
            self._view_indices = []

    @property
    def view_indices(self):
        return tuple(self._view_indices)
    
    def __getitem__(self, location: Any) -> ViewAwareArray:
        view = super().__getitem__(location)
        if view.base is not None:
            # advanced indexing returns a copy, and is not handled here
            # result might be a scalar, which is not a view and has base = None
            view.set_view_location(location)
        return view

    def set_view_location(self, location: Union[Index, Tuple[Index, ...]]) -> None:
        """Update _view_indices with the location just used to create a view.
        Called by __getitem__. If _view_indices is not [], locations must be
        composed. We can assume that the location is well-formed, otherwise
        super().__getitem__ would have thrown an exception.
        """
        if isinstance(location, Tuple):
            location = list(location)
        else:
            location = [location]

        # pad location with None elements until length equals ndim
        # assume Ellipsis only occurs once, since the location is well-formed
        try:
            i = location.index(Ellipsis)
            location[i,i+1] = [None] * (self.ndim - len(location) + 1)
        except ValueError:
            # Ellipsis is not in location
            pass

        # zip the elements of location with the elements of _view_indices that
        # are not integers. Dimensions that have already been indexed with an
        # integer are no longer visible
        j = -1
        for (i, index), (j, new_index) in zip(
            filter(
                lambda i_elem: not isinstance(i_elem[1], int),
                enumerate(self._view_indices)
            ),
            enumerate(location),
        ):
            if index is None:
                # overwrite with new_index
                self._view_indices[i] = new_index
            else: # isinstance(index, slice)
                if isinstance(new_index, int):
                    self._view_indices[i] = slice_plus_int(index, new_index)
                else:  # isinstance(new_index, slice)
                    self._view_indices[i] = slice_plus_slice(index, new_index, self.shape[i])

        # _view_indices were exhausted before location
        # save remaining new indices
        if j + 1 < len(location):
            self._view_indices.extend(location[j+1:])

@numba.njit
def slice_plus_int(s: slice, n: int):
    s = slice(*s.indices(1))
    return s.start + n * s.step

@numba.njit
def slice_plus_slice(s1: slice, s2: slice, list_length: int):
    s1 = slice(*s1.indices(1))
    s2 = slice(*s2.indices(list_length))
    return slice(
        s1.start + s2.start * s1.step,  # start
        s1.start + s2.stop * s1.step,   # stop
        s1.step * s2.step,              # step
    )

if __name__ == "__main__":
    shape = (5,4,3)
    arr = np.arange(np.prod(shape), dtype=np.int32)
    arr = np.reshape(arr, shape)
    arr = ViewAwareArray(arr)
    print("arr.view_indices: ", arr.view_indices)

    scalar = arr[0,0,0]

    advanced_indexing = arr[[0,0,0],[1,1,1]]

    v = arr[1:5]
    print("v.view_indices: ", v.view_indices)
    print("arr.view_indices: ", arr.view_indices)

    v2 = v[2,:3]
    print("v2.view_indices: ", v2.view_indices)
    print("v.view_indices: ", v.view_indices)

    v2_verify = arr[v2.view_indices]
    print(v2)
    print(v2_verify)