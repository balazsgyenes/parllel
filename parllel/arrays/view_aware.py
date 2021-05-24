from copy import deepcopy
from typing import Union, List, Tuple, Any

import numpy as np
from nptyping import NDArray


Index = Union[int, slice, type(Ellipsis), type(np.newaxis)]


class ViewAwareArray(np.ndarray):
    def __new__(cls, input_array: NDArray):
        obj = np.asarray(input_array).view(cls)
        obj._view_locations = []
        return obj

    def __array_finalize__(self, obj):
        # called via explicit constructor
        if obj is None: return

        # called by view casting or new-from-template
        if isinstance(self.base, ViewAwareArray):
            # a view of another ViewAwareArray was created
            # if the array is copied due to advanced indexing, the base will be
            # of type ndarray
            self._view_locations: List[Tuple[Index, ...]] = deepcopy(getattr(obj, "_view_locations", []))
        else:
            self._view_locations = []

    @property
    def view_locations(self):
        #  return copy of list as a tuple
        return tuple(self._view_locations)
    
    def __getitem__(self, location: Any) -> Any:
        result = super().__getitem__(location)
        if isinstance(result.base, ViewAwareArray):
            # result is a view, not a copy
            if not isinstance(location, Tuple):
                location = (location,)
            result._view_locations.append(location)
        return result


# class ViewAwareArray(np.ndarray):
#     def __new__(cls, input_array: NDArray):
#         obj = np.asarray(input_array).view(cls)
#         obj._view_indices = []
#         return obj

#     def __array_finalize__(self, obj):
#         # called via explicit constructor
#         if obj is None: return

#         # called by view casting or new-from-template
#         if self.base is not None:
#             # TODO: verify advanced indexing is always caught by this
#             self._view_indices: List[Union[None, slice, int]] = deepcopy(getattr(obj, "_view_indices", []))
#         else:
#             self._view_indices = []

#     @property
#     def view_indices(self):
#         return tuple(item if item is not None
#                      else slice(None)
#                      for item in self._view_indices)
    
#     def __getitem__(self, location: Any) -> ViewAwareArray:
#         view = super().__getitem__(location)
#         if view.base is not None:
#             # advanced indexing returns a copy, and is not handled here
#             # result might be a scalar, which is not a view and has base = None
#             # TODO: verify advanced indexing is always caught by this
#             view.set_view_location(location)
#         return view

#     def set_view_location(self, location: Union[Index, Tuple[Index, ...]]) -> None:
#         """Update _view_indices with the location just used to create a view.
#         Called by __getitem__. If _view_indices is not [], locations must be
#         composed. We can assume that the location is well-formed, otherwise
#         super().__getitem__ would have thrown an exception.
#         TODO: add support for np.newaxis
#         """
#         if isinstance(location, Tuple):
#             location = list(location)
#         else:
#             location = [location]

#         # pad location with None elements until length equals ndim
#         # assume Ellipsis only occurs once, since the location is well-formed
#         try:
#             i = location.index(Ellipsis)
#             location[i,i+1] = [None] * (self.ndim - len(location) + 1)
#         except ValueError:
#             # Ellipsis is not in location
#             pass

#         # zip the elements of location with the elements of _view_indices that
#         # are not integers. Dimensions that have already been indexed with an
#         # integer are no longer visible
#         # TODO: go back to while loop implementation
#         j = -1
#         for (i, index), (j, new_index) in zip(
#             filter(
#                 lambda i_elem: not isinstance(i_elem[1], int),
#                 enumerate(self._view_indices)
#             ),
#             enumerate(location),
#         ):
#             if index is None:
#                 # overwrite with new_index
#                 self._view_indices[i] = new_index
#             else: # isinstance(index, slice)
#                 if isinstance(new_index, int):
#                     self._view_indices[i] = slice_plus_int(index, new_index)
#                 else:  # isinstance(new_index, slice)
#                     # TODO: self.shape[i] is wrong because the shape of the current view
#                     # might have fewer elements than the shape of the original array
#                     # should it be self.shape[j]?
#                     self._view_indices[i] = slice_plus_slice(index, new_index, self.shape[i])

#         # _view_indices were exhausted before location
#         # save remaining new indices
#         if j + 1 < len(location):
#             self._view_indices.extend(location[j+1:])


# import numba

# @numba.njit
# def slice_plus_int(s: slice, n: int):
#     s = slice(*s.indices(1))
#     return s.start + n * s.step

# @numba.njit
# def slice_plus_slice(s1: slice, s2: slice, list_length: int):
#     s1 = slice(*s1.indices(1))
#     s2 = slice(*s2.indices(list_length))
#     return slice(
#         s1.start + s2.start * s1.step,  # start
#         s1.start + s2.stop * s1.step,   # stop
#         s1.step * s2.step,              # step
#     )

if __name__ == "__main__":
    shape = (10,10,10)
    arr = np.arange(np.prod(shape), dtype=np.int32)
    arr = np.reshape(arr, shape)
    arr = ViewAwareArray(arr)

    test_cases = [
        [
            (slice(1,5)),
            (2, slice(None, 3)),
            (slice(1, 2)),
        ],
        [
            (slice(-1, 1, -2), slice(None, None, -1)),
            (slice(-2, 1, -2), slice(-1, 2, -1)),
        ],
        [
            (Ellipsis, 4),
            (slice(None, 7), slice(1, -1)),
            (3, slice(1, -1)),
        ],
        [
            (2, np.newaxis, np.newaxis, slice(1, -1)),
            (slice(None), slice(None), 4),
        ],
    ]

    for test_case in test_cases:
        view = arr
        for indices in test_case:
            view = view[indices]
        
        assert len(view) > 0

        view_again = arr
        for location in view.view_locations:
            view_again = view_again[location]
        assert np.array_equal(view, view_again), test_case

    # test that indexing a single value does not throw an error
    scalar = arr[(0,) * arr.ndim]

    # test advancing indexing, which produces a copy, does not preserve
    # view locations
    advanced_indexing = arr[[-1,-2,-3], [4,5,6], [1,2,3]]
    assert len(advanced_indexing.view_locations) == 0