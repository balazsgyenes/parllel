# fmt: off
from __future__ import annotations

from typing import Any, Literal, TypeVar

import numpy as np

from parllel import ArrayTree
from parllel.arrays.indices import (Index, Location, add_locations,
                                    batch_dims_from_location, index_slice,
                                    init_location, shape_from_location)
from parllel.arrays.storage import Storage, StorageType

# fmt: on
Self = TypeVar("Self", bound="Array")


class Array:
    """An object wrapping a numpy array for use in sampling. An Array remembers
    indexing operations used to get subarrays. Math operations are generally
    not supported, use `np.asarray(arr)` to get the underlying numpy array.

    Example:
        >>> array = Array(shape=(4, 4, 4), dtype=np.float32)
        >>> array[:, slice(1, 3), 2] = 5.
    """

    _subclasses: dict[str, type[Array]] = {}
    kind = "default"

    def __init_subclass__(
        cls,
        *,
        kind: str | None = None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if kind is None:
            # subclasses can bypass registration machinery by not specifying
            # either kind or storage
            # for example, JaggedArrayList is not a registered subclass
            return
        cls._subclasses[kind] = cls

    @classmethod
    def _get_subclass(cls, kind: str | None) -> type[Array]:
        if kind is None:
            # instantiate a subclass directly by just not passing kind
            # e.g. JaggedArray(shape=(4,4), dtype=np.float32)
            return cls

        # fill in None arguments with values from class used to instantiate
        kind = kind if kind is not None else cls.kind

        if kind == "default":
            # Array is not a registered subclass
            return Array

        # otherwise look up name in dictionary of registered subclasses
        try:
            return cls._subclasses[kind]
        except KeyError:
            raise ValueError(f"No array subclass registered under {kind=}")

    def __new__(
        cls,
        *args,
        kind: str | None = None,
        **kwargs,
    ) -> Array:
        # get requested specialization based on kind/storage
        subcls = cls._get_subclass(kind=kind)
        # give a change for the subclass to specialize itself further based on args/kwargs
        subcls = subcls._specialize_subclass(*args, kind=kind, **kwargs)
        # instantiate that class
        return super().__new__(subcls)

    @classmethod
    def _specialize_subclass(cls, *args, **kwargs) -> type[Array]:
        return cls

    def __init__(
        self,
        batch_shape: tuple[int, ...],
        dtype: np.dtype,
        *,
        feature_shape: tuple[int, ...] = (),
        kind: str | None = None,  # consumed by __new__
        storage: StorageType = "local",
        padding: int = 0,
        full_size: int | None = None,
    ) -> None:
        if not batch_shape:
            raise ValueError("Non-empty batch_shape required.")

        dtype = np.dtype(dtype)
        if dtype == np.object_:
            raise ValueError("Data type should not be object.")

        if padding < 0:
            raise ValueError("Padding must be non-negative.")
        if padding > batch_shape[0]:
            raise ValueError(
                f"Padding ({padding}) cannot be greater than leading dimension {batch_shape[0]}."
            )

        if full_size is None:
            full_size = batch_shape[0]
        if full_size < batch_shape[0]:
            raise ValueError(
                f"Full size ({full_size}) cannot be less than leading dimension {batch_shape[0]}."
            )
        if full_size % batch_shape[0] != 0:
            raise ValueError(
                f"Full size ({full_size}) must be evenly divided by leading dimension {batch_shape[0]}."
            )

        self.dtype = dtype
        self.padding = padding
        # size of leading dim of full array, without padding
        self.full_size = full_size
        self._base_shape = (full_size + 2 * padding,) + batch_shape[1:] + feature_shape
        self._base_batch_dims = len(batch_shape)

        self._storage = Storage(
            kind=storage,
            shape=self._base_shape,
            dtype=dtype,
        )

        self._current_location = init_location(self._base_shape)
        init_slice = slice(padding, batch_shape[0] + padding)
        self._unresolved_indices: list[Location] = [init_slice]
        self._resolve_indexing_history()

        self._rotatable = True

    def new_array(
        self,
        batch_shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        *,
        feature_shape: tuple[int, ...] | None = None,
        kind: str | None = None,
        storage: StorageType | None = None,
        padding: int | None = None,
        full_size: int | None = None,
        inherit_full_size: bool = False,
        **kwargs,
    ) -> Array:
        """Creates an Array with the same shape and type as a given Array
        (similar to torch's new_zeros function). By default, the full size of
        the created Array is just the apparent size of the template. To set it
        to another value, either pass it manually or set the
        `inherit_full_size` flag to True to use the template's full size.
        """
        batch_shape = batch_shape if batch_shape is not None else self.batch_shape
        feature_shape = (
            feature_shape
            if feature_shape is not None
            else self.shape[self.n_batch_dims :]
        )
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
                self.full_size
                if (
                    inherit_full_size
                    and self._rotatable
                    and self.full_size > self.shape[0]
                )
                else None
            )
        )
        # We use Array() because using type(self)() would skip calling __init__
        # if the new array is not a subclass of the current array
        # (e.g. shared_mem_array.new_array(storage="local"))
        return Array(
            feature_shape=feature_shape,
            dtype=dtype,
            batch_shape=batch_shape,
            kind=kind,
            storage=storage,
            padding=padding,
            full_size=full_size,
            **kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        example: Any,
        *,
        force_32bit: Literal[True, "float", "int", False] = True,
        **kwargs,
    ) -> Array:
        subcls = cls._get_subclass(kind=kwargs.get("kind"))
        kwargs = subcls._get_from_numpy_kwargs(example, kwargs)

        np_example: np.ndarray = np.asanyarray(example)  # promote scalars to 0d arrays

        if kwargs.get("feature_shape") is None:
            kwargs["feature_shape"] = np_example.shape

        if kwargs.get("dtype") is None:
            dtype = np_example.dtype
            if dtype == np.int64 and force_32bit in {True, "int"}:
                dtype = np.int32
            elif dtype == np.float64 and force_32bit in {True, "float"}:
                dtype = np.float32
            kwargs["dtype"] = dtype

        return cls(**kwargs)

    @classmethod
    def _get_from_numpy_kwargs(cls, example: Any, kwargs: dict) -> dict:
        return kwargs

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            self._resolve_indexing_history()

        return self._shape

    @property
    def storage(self) -> StorageType:
        return self._storage.kind

    @property
    def n_batch_dims(self) -> int:
        if self._shape is None:
            self._resolve_indexing_history()

        return self._n_batch_dims

    @property
    def batch_shape(self) -> tuple[int, ...]:
        if self._shape is None:
            self._resolve_indexing_history()

        return self._batch_shape

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

    def resize(
        self,
        feature_shape: tuple[int, ...] | None = None,
        batch_shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        padding: int | None = None,
        full_size: int | None = None,
    ) -> None:
        # also catches case if batch_shape == ()
        batch_shape = batch_shape if batch_shape else self.batch_shape
        feature_shape = (
            feature_shape
            if feature_shape is not None
            else self.shape[self.n_batch_dims :]
        )
        dtype = dtype or self.dtype
        padding = padding if padding is not None else self.padding
        full_size = full_size if full_size is not None else self.full_size

        self.dtype = dtype
        self.padding = padding
        self.full_size = full_size
        self._base_shape = (full_size + 2 * padding,) + batch_shape[1:] + feature_shape
        self._base_batch_dims = len(batch_shape)
        # TODO: what to do about all the edge cases where subarrays calls
        # resize()?
        # should base_shape and dtype be part of storage?
        self._storage.resize(shape=self._base_shape, dtype=dtype)

    def __getitem__(self: Self, indices: Location) -> Self:
        # new Array object initialized through a (shallow) copy. Attributes
        # that differ between self and result are modified next. This allows
        # subclasses to override and only handle additional attributes that
        # need to be modified.
        subarray: Self = object.__new__(type(self))
        subarray.__dict__.update(self.__dict__)
        # disallow rotate and reset on subarrays
        subarray._rotatable = False
        # assign *copy* of _unresolved_indices with additional element for this
        # indexing operation
        subarray._unresolved_indices = subarray._unresolved_indices + [indices]
        # set shape to None to indicate that indexing must be resolved
        subarray._shape = None
        return subarray

    def _resolve_indexing_history(self) -> None:
        for location in self._unresolved_indices:
            self._current_location = add_locations(
                self._current_location,
                location,
                self._base_shape,
                neg_from_end=False,
            )

        self._unresolved_indices.clear()

        self._shape = shape_from_location(self._current_location, self._base_shape)
        self._n_batch_dims = batch_dims_from_location(
            self._current_location, self._base_batch_dims
        )
        self._batch_shape = self._shape[: self._n_batch_dims]

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
        with self._storage as base_array:
            base_array[destination] = value

    @property
    def full(self: Self) -> Self:
        # no need to resolve indexing history since we clear it

        full: Self = self.__new__(type(self))
        full.__dict__.update(self.__dict__)
        full._rotatable = False

        # clear any unresolved indexing history
        full._unresolved_indices.clear()

        # assign current location so that full array except padding is visible
        full._current_location = init_location(full._base_shape)
        init_slice = slice(full.padding, full.full_size + full.padding)
        full._unresolved_indices.append(init_slice)
        full._resolve_indexing_history()

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

        offsetted: Self = self.__new__(type(self))
        offsetted.__dict__.update(self.__dict__)
        offsetted._rotatable = False
        offsetted._current_location = list(self._current_location)

        leading_loc = offsetted._current_location[0]
        if isinstance(leading_loc, slice):
            offset_slice = slice(offset, offsetted.shape[0] + offset)
            offsetted._current_location[0] = index_slice(
                leading_loc,
                offset_slice,
                offsetted._base_shape[0],
                neg_from_end=False,
            )
        else:
            offsetted._current_location[0] = leading_loc + offset

        offsetted._resolve_indexing_history()
        return offsetted

    def reset(self) -> None:
        """Resets array, such that the offset will be 0 after the next time
        that `rotate()` is called. This is useful in the sampler, which calls
        `rotate()` before every batch.
        """
        if not self._rotatable:
            raise ValueError("reset() is only allowed on unindexed array.")

        self._current_location[0] = slice(
            self.full_size + self.padding - self.shape[0],
            self.full_size + self.padding,
            1,
        )

    def rotate(self) -> None:
        if not self._rotatable:
            raise ValueError("rotate() is only allowed on unindexed array.")

        leading_loc: slice = self._current_location[0]
        start = leading_loc.start + self.shape[0]
        stop = leading_loc.stop + self.shape[0]

        if start >= self.full_size:
            # wrap around to beginning of array
            start -= self.full_size
            stop -= self.full_size

            if self.padding:
                # copy values from end of base array to beginning
                final_values = slice(
                    self.full_size - self.padding,
                    self.full_size + self.padding,
                )
                next_previous_values = slice(-self.padding, self.padding)
                full_array = self.full
                full_array[next_previous_values] = full_array[final_values]

        # update current location with modified start/stop
        self._current_location[0] = slice(start, stop, 1)

    def to_ndarray(self) -> ArrayTree[np.ndarray]:
        if self._shape is None:
            self._resolve_indexing_history()

        base_array = self._storage.get_numpy()
        return base_array[tuple(self._current_location)]

    def __array__(self, dtype=None) -> np.ndarray:
        array = self.to_ndarray()
        array = np.asarray(array)  # promote scalars to 0d arrays
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

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
        self._storage.close(force=True)


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
