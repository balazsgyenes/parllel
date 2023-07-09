from __future__ import annotations

from typing import Any, Literal, Optional, TypeVar, Union

import numpy as np

from parllel.arrays.indices import (Index, Location, add_locations,
                                    init_location, shape_from_location)

Self = TypeVar("Self", bound="Array")


class Array:
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
    _base_array: np.ndarray

    def __init_subclass__(
        cls,
        *,
        kind: Optional[str] = None,
        storage: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)
        kind = kind if kind is not None else "default"
        storage = storage if storage is not None else "local"
        cls._subclasses[(kind, storage)] = cls

    def __new__(
        cls,
        *args,
        kind: Optional[str] = None,
        storage: Optional[str] = None,
        **kwargs,
    ) -> Array:
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
            raise ValueError(
                f"No array subclass registered under {kind=} and {storage=}"
            )
        return super().__new__(subcls)

    def __init__(
        self,
        shape: tuple[int, ...],
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
        # size of leading dim of full array, without padding
        self._full_size = full_size
        base_T_size = full_size + 2 * padding
        self._base_shape = self._allocate_shape = (base_T_size,) + shape[1:]

        self._allocate(shape=self._base_shape, dtype=dtype, name="_base_array")

        self._current_location = init_location(self._base_shape)
        init_slice = slice(padding, shape[0] + padding)
        self._current_location = add_locations(
            self._current_location,
            init_slice,
            self._base_shape,
            neg_from_end=False,
        )
        self._index_history: list[Location] = []

        self._shape = shape  # shape of current array
        self._rotatable = True

    def new_array(
        self,
        shape: Optional[tuple[int, ...]] = None,
        dtype: Optional[np.dtype] = None,
        kind: Optional[str] = None,
        storage: Optional[str] = None,
        padding: Optional[int] = None,
        full_size: Optional[int] = None,
        inherit_full_size: bool = False,
    ) -> Array:
        """Creates an Array with the same shape and type as a given Array
        (similar to torch's new_zeros function). By default, the full size of
        the created Array is just the apparent size of the template. To set it
        to another value, either pass it manually or set the
        `inherit_full_size` flag to True to use the template's full size.
        """
        shape = shape if shape is not None else self.shape
        dtype = dtype or self.dtype
        kind = kind if kind is not None else self.kind
        storage = storage if storage is not None else self.storage
        padding = padding if padding is not None else self.padding
        full_size = (
            full_size
            if full_size is not None
            else (
                # only inherit full_size from self if the user explicitly
                # requests it, the array has not been indexed, and full_size
                # is not the default
                self._full_size
                if (
                    inherit_full_size
                    and self._rotatable
                    and self._full_size > self.shape[0]
                )
                else None
            )
        )
        return type(self)(
            shape,
            dtype,
            kind=kind,
            storage=storage,
            padding=padding,
            full_size=full_size,
        )

    @classmethod
    def from_numpy(
        cls,
        example: Any,
        shape: Optional[tuple[int, ...]] = None,
        dtype: Optional[np.dtype] = None,
        force_32bit: Literal[True, "float", "int", False] = True,
        kind: Optional[str] = None,
        storage: Optional[str] = None,
        padding: int = 0,
        full_size: Optional[int] = None,
    ) -> Array:
        np_example = np.asanyarray(example)  # promote scalars to 0d arrays
        if dtype is None:
            dtype = np_example.dtype
            if dtype == np.int64 and force_32bit in {True, "int"}:
                dtype = np.int32
            elif dtype == np.float64 and force_32bit in {True, "float"}:
                dtype = np.float32
        return cls(
            # TODO: replace shape with feature_shape
            shape=shape if shape is not None else np_example.shape,
            dtype=dtype,
            kind=kind,
            storage=storage,
            padding=padding,
            full_size=full_size,
        )

    def _allocate(self, shape: tuple[int, ...], dtype: np.dtype, name: str) -> None:
        # initialize numpy array
        setattr(self, name, np.zeros(shape=shape, dtype=dtype))

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            self._resolve_indexing_history()

        return self._shape

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
        if self._shape is None:
            self._resolve_indexing_history()

        return self._shape[0] - 1

    def __getitem__(self: Self, indices: Location) -> Self:
        # new Array object initialized through a (shallow) copy. Attributes
        # that differ between self and result are modified next. This allows
        # subclasses to override and only handle additional attributes that
        # need to be modified.
        subarray: Self = self.__new__(type(self))
        subarray.__dict__.update(self.__dict__)
        # disallow rotate and reset on subarrays
        subarray._rotatable = False
        # assign *copy* of _index_history with additional element for this
        # indexing operation
        subarray._index_history = subarray._index_history + [indices]
        # set shape to None to indicate that indexing must be resolved
        subarray._shape = None
        return subarray

    def _resolve_indexing_history(self) -> None:
        for location in self._index_history:
            self._current_location = add_locations(
                self._current_location,
                location,
                self._base_shape,
                neg_from_end=False,
            )

        self._index_history.clear()

        self._shape = shape_from_location(self._current_location, self._base_shape)

    def __setitem__(self, indices: Location, value: Any) -> None:
        if self._shape is None:
            self._resolve_indexing_history()

        destination = tuple(
            add_locations(
                self._current_location,
                indices,
                self._base_shape,
                neg_from_end=False,
            )
        )

        # write to underlying array at that location
        self._base_array[destination] = value

    @property
    def full(self: Self) -> Self:
        # no need to resolve indexing history since we clear it

        full: Self = self.__new__(type(self))
        full.__dict__.update(self.__dict__)
        full._rotatable = False

        # clear any unresolved indexing history
        full._index_history.clear()

        # assign current location so that full array except padding is visible
        full._current_location = init_location(full._base_shape)
        init_slice = slice(full.padding, full._full_size + full.padding)
        full._current_location = add_locations(
            full._current_location,
            init_slice,
            full._base_shape,
            neg_from_end=False,
        )

        # explicitly compute shape, since we know what it will be
        full._shape = (full._full_size,) + full._base_shape[1:]

        return full

    @property
    def next(self: Self) -> Self:
        return self._get_at_offset(offset=1)

    @property
    def previous(self: Self) -> Self:
        return self._get_at_offset(offset=-1)

    def _get_at_offset(self: Self, offset: int) -> Self:
        if self._shape is None:
            self._resolve_indexing_history()

        offsetted: Array = self.__new__(type(self))
        offsetted.__dict__.update(self.__dict__)
        offsetted._rotatable = False

        leading_loc = offsetted._current_location[0]
        if isinstance(leading_loc, slice):
            offset_slice = slice(offset, offsetted.shape[0] + offset)
            offsetted._current_location = add_locations(
                offsetted._current_location,
                offset_slice,
                offsetted._base_shape,
                neg_from_end=False,
            )
        else:
            offsetted._current_location[0] = leading_loc + offset

        return offsetted

    def reset(self) -> None:
        """Resets array, such that the offset will be 0 after the next time
        that `rotate()` is called. This is useful in the sampler, which calls
        `rotate()` before every batch.
        """
        if not self._rotatable:
            raise ValueError("reset() is only allowed on unindexed array.")

        self._current_location[0] = slice(
            self._full_size + self.padding - self.shape[0],
            self._full_size + self.padding,
            1,
        )

    def rotate(self) -> None:
        if not self._rotatable:
            raise ValueError("rotate() is only allowed on unindexed array.")

        leading_loc = self._current_location[0]
        start = leading_loc.start
        stop = leading_loc.stop

        start += self.shape[0]
        if wrap_bounds := start >= self._full_size:
            # wrap both start and stop simultaneously
            start %= self._full_size
            stop %= self._full_size
        # mod and increment stop in opposite order
        # because stop - start = self.shape[0], we can be sure that stop will
        # be reduced by mod even before incrementing
        stop += self.shape[0]

        self._current_location[0] = slice(start, stop, 1)

        if self.padding and wrap_bounds:
            # copy values from end of base array to beginning
            final_values = slice(
                self._full_size - self.padding,
                self._full_size + self.padding,
            )
            next_previous_values = slice(-self.padding, self.padding)
            self.full[next_previous_values] = self.full[final_values]

    def __array__(self, dtype=None) -> np.ndarray:
        if self._shape is None:
            self._resolve_indexing_history()

        array = self._base_array[tuple(self._current_location)]
        array = np.asarray(array)  # promote scalars to 0d arrays
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def to_tree(self) -> Union[np.ndarray, dict[str, np.ndarray]]:
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
            prefix
            + np.array2string(
                self.__array__(),
                separator=",",
                prefix=prefix,
                suffix=suffix,
            )
            + suffix
        )

    def __bool__(self) -> bool:
        return bool(self.__array__())

    def __eq__(self, o: object) -> np.ndarray:
        return self.__array__() == o

    def close(self):
        pass


def shift_indices(
    indices: Location,
    shift: int,
    size: int,
) -> tuple[Index, ...]:
    if isinstance(indices, tuple):
        first, rest = indices[0], indices[1:]
    else:
        first, rest = indices, ()
    return shift_index(first, shift, size) + rest


def shift_index(
    index: Index,
    shift: int,
    size: int,
) -> tuple[Index, ...]:
    """Shifts an array index up by an integer value."""
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
