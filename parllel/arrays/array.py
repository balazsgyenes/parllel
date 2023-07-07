from __future__ import annotations  # full returns another Array

from functools import reduce
from typing import Any, Optional, Tuple, TypeVar

import numpy as np

from parllel.buffers import Buffer, Index, Indices

Self = TypeVar("Self", bound="Array")


class Array(Buffer):
    """An object wrapping a numpy array for use in sampling. An Array remembers
    indexing operations used to get subarrays. Math operations are generally
    not supported, use `np.asarray(arr)` to get the underlying numpy array.

    Example:
        >>> array = Array(shape=(4, 4, 4), dtype=np.float32)
        >>> array[:, slice(1, 3), 2] = 5.
    """
    _subclasses = {}
    storage = "local"
    kind = "default"

    def __init_subclass__(cls, /, kind: Optional[str] = None, storage: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        kind = kind if kind is not None else "default"
        storage = storage if storage is not None else "local"
        cls._subclasses[(kind, storage)] = cls

    def __new__(cls, *args, kind: Optional[str] = None, storage: Optional[str] = None, **kwargs):
        # fill in empty arguments with values from class used to instantiate
        # can instantiate a subclass directly by just not passing kind/storage
        # e.g. SharedMemoryArray(shape=(4,4), dtype=np.float32)
        kind = kind if kind is not None else cls.kind
        storage = storage if storage is not None else cls.storage

        if kind == "default" and storage == "local":
            # instantiating "Array" with default arguments
            return super().__new__(cls)
        # otherwise look up name in dictionary of registered subclasses
        try:
            subcls = cls._subclasses[(kind, storage)]
        except KeyError:
            raise ValueError(f"No array registered under storage type '{storage}'")
        return super().__new__(subcls)

    def __init__(self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        *,
        kind: Optional[str] = None,  # consumed by __new__
        storage: Optional[str] = None,  # consumed by __new__
        padding: int = 0,
        full_size: Optional[int] = None,
    ) -> None:

        if not shape:
            raise ValueError("Non-empty shape required.")

        dtype = np.dtype(dtype)
        if dtype == np.object_:
            raise ValueError("Data type should not be object.")

        if padding < 0:
            raise ValueError("Padding must be non-negative.")
        if padding > shape[0]:
            raise ValueError(
                f"Padding ({padding}) cannot be greater than leading "
                f"dimension {shape[0]}."
            )

        if full_size is None:
            full_size = shape[0]
        if full_size < shape[0]:
            raise ValueError(
                f"Full size ({full_size}) cannot be less than "
                f"leading dimension {shape[0]}."
            )
        if full_size % shape[0] != 0:
            raise ValueError(
                f"The leading dimension {shape[0]} must divide the full "
                f"size ({full_size}) evenly."
            )

        self.dtype = dtype
        self.padding = padding

        self._default_size = shape[0]  # size of leading dim of unindexed array
        self._full_size = full_size  # size of leading dim of full array, without padding
        self._offset = 0  # offset of visible region within full array, without padding
        self._shift = self._offset + padding  # offset of visible region within base array
        # add padding onto both ends of first dimension
        self._base_shape = (full_size + 2 * padding,) + shape[1:]

        self._buffer_id: int = id(self)
        self._index_history: list[Indices] = []

        self._allocate()

        self._current_array = self._base_array  # the result of np.asarray()
        self._apparent_shape = shape  # shape of current array

        # used to enable indexing into a single element like element[:] = 0
        # set to the previous value of current_array, or the base_array
        self._previous_array = self._base_array

        self._resolve_indexing_history()

    @classmethod
    def like(cls,
        template: Array,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[np.dtype] = None,
        kind: Optional[str] = None,
        storage: Optional[str] = None,
        padding: Optional[int] = None,
        full_size: Optional[int] = None,
        inherit_full_size: bool = False,
    ) -> Array:
        """Creates an Array with the same shape and type as a given Array
        (similar to numpy's zeros_like function). By default, the full size of
        the created Array is just the apparent size of the template. To set it
        to another value, either pass it manually or set the
        `inherit_full_size` flag to True to use the template's full size.

        # TODO: turn this into a normal method called new_array
        """
        shape = shape if shape is not None else template.shape
        dtype = dtype or template.dtype
        kind = kind or template.kind
        storage = storage or template.storage
        padding = padding if padding is not None else template.padding
        # only inherit full_size from template if full_size is not the default and flag set to true
        full_size = full_size if full_size is not None else (
            template._full_size
            if (inherit_full_size and template._full_size > template._default_size) else
            None
        )
        return cls(
            shape,
            dtype,
            kind=kind,
            storage=storage,
            padding=padding,
            full_size=full_size,
        )

    def _allocate(self) -> None:
        # initialize numpy array
        self._base_array = np.zeros(shape=self._base_shape, dtype=self.dtype)

    @property
    def shape(self):
        if self._apparent_shape is None:
            self._resolve_indexing_history()

        return self._apparent_shape
        
    @property
    def first(self) -> int:
        """The index of the first element in the array, not including padding.
        Enables syntactic sugar like `arr[arr.first - 1]`
        """
        return 0

    @property
    def last(self) -> int:
        """The index of the final element in the array, not including padding.
        Replaces indexing at -1 in numpy arrays.
        e.g. array[-1] -> rot_array[rot_array.last]
        """
        if self._apparent_shape is None:
            self._resolve_indexing_history()
        
        return self._apparent_shape[0] - 1

    def __getitem__(self, location: Indices) -> Self:
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

    def _resolve_indexing_history(self) -> None:
        array = self._base_array
        self._shift = shift = self._offset + self.padding

        if self._index_history:
            # shift only the first indices, leave the rest (if there are more)
            index_history = [shift_indices(
                self._index_history[0],
                shift,
                self._default_size,
            )] + self._index_history[1:]
        else:
            # even if the array was never indexed, only this slice of the array
            # should be returned by __array__
            index_history = [slice(shift, shift + self._default_size)]

        # if index history has only 1 element, this has no effect
        array = reduce(lambda arr, index: arr[index], index_history[:-1], array)
        self._previous_array = array
        
        # we guarantee that index_history has at least 1 element
        array = array[index_history[-1]]

        self._current_array = array
        self._apparent_shape = array.shape
    
    def __setitem__(self, location: Indices, value: Any) -> None:
        # TODO: optimize this method to avoid resolving history if location
        # is slice(None) or Ellipsis and history only has one element
        # in this case, only previous_array is required
        if self._current_array is None:
            self._resolve_indexing_history()

        if self._index_history:
            if self._apparent_shape == ():
                # Need to avoid item assignment on a scalar (0-D) array, so we assign
                # into previous array using the last indices used
                if not (location == slice(None) or location == ...):
                    raise IndexError("Cannot take slice of 0-D array.")
                location = self._index_history[-1]
                # indices must be shifted if they were the first indices
                if len(self._index_history) == 1:
                    location = shift_indices(location, self._shift, self._default_size)
                destination = self._previous_array
            else:
                destination = self._current_array
        else:
            location = shift_indices(location, self._shift, self._default_size)
            destination = self._base_array
        destination[location] = value
    
    @property
    def full(self) -> Array:
        full: Array = self.__new__(type(self))
        full.__dict__.update(self.__dict__)

        full._default_size = full._full_size
        full._offset = 0

        full._index_history = []
        full._current_array = None
        full._apparent_shape = None
        return full

    @property
    def next(self) -> Array:
        return self._get_at_offset(offset=1)

    @property
    def previous(self) -> Array:
        return self._get_at_offset(offset=-1)

    def _get_at_offset(self, offset: int) -> Array:
        if self._index_history:
            raise RuntimeError("Only allowed to get at offset from unindexed array")

        new: Array = self.__new__(type(self))
        new.__dict__.update(self.__dict__)

        # total shift of offset cannot exceed +/-padding, but we do not check
        # if padding is exceeded, getitem/setitem may throw error
        new._offset += offset

        # index_history is already empty
        # current array is now invalid, but apparent shape should still be
        # correct
        new._current_array = None
        return new

    def reset(self) -> None:
        """Resets array, such that the offset will be 0 after the next time
        that `rotate()` is called. This is useful in the sampler, which calls
        `rotate()` before every batch.
        """
        if self._index_history:
            raise RuntimeError("Only allowed to call `reset()` on original array")
        
        # if apparent size is not smaller, sets offset to 0
        self._offset = self._full_size - self._default_size

        # current array is now invalid, but apparent shape should still be
        # correct
        self._current_array = None

    def rotate(self) -> None:

        if self._index_history:
            raise RuntimeError("Only allowed to call `rotate()` on original array")

        self._offset += self._default_size

        if self.padding and self._offset >= self._full_size:
            # copy values from end of base array to beginning
            final_values = slice(
                self._full_size - self.padding,
                self._full_size + self.padding,
            )
            next_previous_values = slice(-self.padding, self.padding)
            self.full[next_previous_values] = self.full[final_values]

        self._offset %= self._full_size

        # current array is now invalid, but apparent shape should still be
        # correct
        self._current_array = None

    def __array__(self, dtype=None) -> np.ndarray:
        if self._current_array is None:
            self._resolve_indexing_history()
        
        array = np.asarray(self._current_array)  # promote scalars to 0d arrays
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def __buffer__(self) -> Buffer:
        return self.__array__()

    def __repr__(self) -> str:
        prefix = type(self).__name__ + "("
        suffix = (
            f", storage={self.storage}"
            f", dtype={self.dtype.name}"
            f", padding={self.padding}"
            ")"
        )
        return (
            prefix +
            np.array2string(
                self.__array__(),
                separator=",",
                prefix=prefix,
                suffix=suffix
            ) +
            suffix
        )

    def __bool__(self) -> bool:
        return bool(self.__array__())

    def __eq__(self, o: object) -> np.ndarray:
        return self.__array__() == o

    def close(self):
        pass


def shift_indices(indices: Indices, shift: int, size: int,
) -> Tuple[Index, ...]:
    if isinstance(indices, tuple):
        first, rest = indices[0], indices[1:]
    else:
        first, rest = indices, ()
    return shift_index(first, shift, size) + rest


def shift_index(index: Index, shift: int, size: int,
) -> Tuple[Index, ...]:
    """Shifts an array index up by an integer value.
    """
    if isinstance(index, int):
        if index < -shift:
            raise IndexError(
                f"Not enough padding ({shift}) to accomodate index ({index})"
            )
        return (index + shift,)
    if isinstance(index, np.ndarray):
        if np.any(index < -shift):
            raise IndexError(
                f"Not enough padding ({shift}) to accomodate index ({index})"
            )
        return (index + shift,)
    if isinstance(index, slice):
        flipped = index.step is not None and index.step < 0
        # in case the step is negative, we need to reverse/adjust the limits
        # limits must be incremented because the upper limit of the slice is
        # not in the slice
        # [:] = slice(None, None, None) -> slice(shift, shift+size, None)
        # [::-1] = slice(None, None, -1) -> slice(shift+size-1, shift-1, -1)
        # [:3:-1] = slice(None, 3, -1) -> slice(shift+size-1, 3+shift, -1)

        if index.start is not None:
            start = index.start + shift
        else:
            start = shift + size - 1 if flipped else shift
        
        if index.stop is not None:
            stop = index.stop + shift
        else:
            stop = shift - 1 if flipped else shift + size
            # only way to represent slice that ends at beginning of array is
            # stop=None, since e.g. stop=-1 means stop at the last element
            stop = stop if stop >= 0 else None

        return (slice(start, stop, index.step),)
    if index is Ellipsis:
        # add another Ellipsis, to index any remaining dimensions that an
        # Ellipsis would have indexed (possible no extra dimensions remain)
        return (slice(shift, shift + size), Ellipsis)
    raise ValueError(index)
