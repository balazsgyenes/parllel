import math
from typing import TypeVar, Union

import numpy as np

# A single index element, e.g. arr[3:6]
Index = Union[int, slice, np.ndarray, type(Ellipsis), list[int]]
StandardIndex = Union[int, slice, np.ndarray]
BasicIndex = Union[int, slice]

# A single indexing location, e.g. arr[2, 0] or arr[:-2]
Location = Union[Index, tuple[Index, ...]]
StandardLocation = list[StandardIndex]

Shape = tuple[int, ...]


def compute_indices(
    base_array: np.ndarray,
    current_array: np.ndarray,
) -> list[BasicIndex]:
    current_pointer = current_array.__array_interface__["data"][0]
    base_pointer = base_array.__array_interface__["data"][0]
    offset = current_pointer - base_pointer

    # total offset is a linear combination of base strides
    # get the coefficients of that linear combination
    # this are either the start of the slices for that dimension, or the index
    # itself if indexed with an integer
    base_strides = base_array.strides
    dim_offsets = [0 for _ in base_strides]
    for dim, base_stride in enumerate(base_strides):
        dim_offsets[dim] = offset // base_stride
        offset %= base_stride
    assert offset == 0

    base_shape = base_array.shape
    base_strides = base_array.strides
    curr_shape = current_array.shape
    curr_strides = current_array.strides
    current_indices = [None for _ in base_shape]

    for curr_size, curr_stride in zip(curr_shape, curr_strides):
        # search for the corresponding stride in base_array
        # base_stride_(dim-1) > abs(current_stride_(dim)) >= base_stride_(dim)
        # the absolute value is required because the current stride might be negative
        dim = next(
            index for index, elem in enumerate(base_strides) if abs(curr_stride) >= elem
        )
        base_stride = base_strides[dim]

        step = curr_stride // base_stride
        start = dim_offsets[dim]
        stop = start + curr_size * step
        if step < 1 and stop == step:
            # for negative indices, stop=-1 means stop just after 0
            # unfortunately, this is ambiguous, since -1 is the last element
            stop = None

        step = step if step != 1 else None

        current_indices[dim] = slice(start, stop, step)

    for dim in range(len(base_shape)):
        if current_indices[dim] is None:
            current_indices[dim] = dim_offsets[dim]

    return current_indices


def predict_copy_on_index(shape: Shape, new_location: Location) -> bool:
    if not isinstance(new_location, tuple):
        new_location = (new_location,)
    return len(new_location) == len(shape) and all(
        isinstance(index, int) for index in new_location
    )


def add_locations(
    location: StandardLocation,
    new_location: Location,
    base_shape: Shape,
    neg_from_end: bool = True,
) -> StandardLocation:
    """Takes an indexing location, the current indices of the subarray relative
    to the base array, and the base array's shape, and returns the next indices
    of the subarray relative to the base array after indexing.

    If `neg_from_end` is True (default), negative indices are interpreted as
    relative to the end of the array (as standard in Python and numpy). If
    False, negative indices are relative to the start of the array, allowing
    access to the base array from within a subarray.

    If location contains slices, they must be in standard form (see
    `clean_slice`).
    """
    if any(isinstance(location, np.ndarray) for location in location):
        raise IndexError(
            "Cannot processing further indexing operations after advanced indexing."
        )

    location = list(location)  # create a copy to prevent modifying inputs

    if isinstance(new_location, tuple):
        new_location = list(new_location)
    else:
        new_location = [new_location]

    # check if Ellipsis occurs in new_location
    ellipses = [
        dim for dim, new_index in enumerate(new_location) if new_index is Ellipsis
    ]
    if ellipses:
        if len(ellipses) > 1:
            raise IndexError("An index can only have a single ellipsis ('...').")
        # pad new_location with slice(None) elements until length equals the
        # apparent number of dimensions
        i = ellipses[0]
        apparent_n_dim = len(
            [location for location in location if isinstance(location, slice)]
        )
        new_location[i : i + 1] = [slice(None)] * (
            apparent_n_dim - len(new_location) + 1
        )

    if not new_location:
        # e.g. if new_location was [...] or [()]
        return location

    i = 0
    for dim, (index, size) in enumerate(zip(location, base_shape)):
        if isinstance(index, int):
            # this dimension has already been indexed with an integer, it
            # cannot be indexed further
            continue

        location[dim] = index_slice(index, new_location[i], size, neg_from_end)

        i += 1  # consider next new_index on next loop iteration
        if i == len(new_location):
            # new_location exhausted
            break
    else:
        # if new_location has not been exhausted, too many indices
        raise IndexError(
            f"Too many indices for array: array is {len(base_shape)}-dimensional, "
            f"but {len(new_location)} were indexed."
        )

    return location


IndexType = TypeVar("IndexType", int, slice, np.ndarray)


def index_slice(
    index: slice,
    new_index: IndexType,
    size: int,
    neg_from_end: bool = True,
) -> IndexType:
    """Takes the current indexing state of a single dimension and a new index,
    and returns a single index (int or slice) that could be used to index the
    base array to achieve the same result.

    index must be a slice in standard form (see `clean_slice`).
    """
    if isinstance(new_index, np.ndarray):
        new_index = index_slice_with_np(index, new_index, size, neg_from_end)
        if not np.all((0 <= new_index) & (new_index < size)):
            out_of_bounds = new_index[~(0 <= new_index)]
            if out_of_bounds.shape[0] == 0:
                out_of_bounds = new_index[~(new_index < size)]
            raise IndexError(
                f"Index {out_of_bounds[0]} is out of bounds for axis with size "
                f"{size}."
            )
        return new_index

    elif isinstance(new_index, int):
        new_index = index_slice_with_int(index, new_index, size, neg_from_end)
        if not (0 <= new_index < size):
            raise IndexError(
                f"Index {new_index} is out of bounds for axis with size {size}."
            )
        return new_index  # new_index should never be negative after cleaning

    elif isinstance(new_index, slice):
        if new_index == slice(None):
            # no op if indexed with the trivial slice
            return index

        else:
            # convert to numerical form for computation
            start, stop, step = index.indices(size)

            new_step = new_index.step if new_index.step is not None else 1

            if (new_start := new_index.start) is not None:
                # translate new_index.start into "world coordinates"
                new_start = index_slice_with_int(index, new_start, size, neg_from_end)
                # lower bound at 0, preventing out-of-bounds slices from turning
                # into in-bounds slices
                new_start = max(0, new_start)
            elif new_step > 0:
                new_start = start
            else:
                # compute true end point of slice, taking into account that step
                # might not evenly divide the length of the array
                new_start = start + step * math.ceil((stop - start) / step) - step

            if (new_stop := new_index.stop) is not None:
                # translate new_stop into "world coordinates"
                new_stop = index_slice_with_int(index, new_stop, size, neg_from_end)
                if neg_from_end:
                    # find the stop that results in the shorter sequence
                    new_stop = min(stop, new_stop) if step > 0 else max(stop, new_stop)
                # lower bound at 0 only if positive step. negative stops for
                # negative steps are converted into Nones later
                new_stop = max(0, new_stop) if step > 0 else new_stop
            elif new_step > 0:
                new_stop = stop
            else:
                new_stop = start
                # extend new_stop one step away from body of range
                new_stop += step * np.sign(new_step)

            # update step by multiplying new step onto it
            new_step = step * new_step

            # set to None instead of -1
            # this might happen if the stop was specified as an integer, but lands
            # before the beginning of the array, or if the stop was None and the
            # start/stop of index was -1 (i.e. None with negative step)
            new_stop = None if new_stop < 0 else new_stop

            return slice(new_start, new_stop, new_step)

    else:
        raise TypeError(f"Cannot index slice with {type(new_index).__name__} object")


def index_slice_with_int(
    slice_: slice, index: int, size: int, neg_from_end: bool = True
) -> int:
    """Compute the index of the element that would be returned if a vector was
    first indexed with a slice and then an integer. The slice must be given in
    standard form.
    """
    step = slice_.step
    if index >= 0 or not neg_from_end:
        zero = slice_.start
    else:
        # if the index is negative, the start is the end of the slice, or the
        # end of the array
        start, stop, step = slice_.indices(size)

        # correction if step does not evenly divide into size of array
        # find effective end of slice by counting from start of slice
        zero = start + step * math.ceil((stop - start) / step)

    return zero + index * step


def index_slice_with_np(
    slice_: slice, index: np.ndarray, size: int, neg_from_end: bool = True
) -> np.ndarray:
    """Compute the index of the element that would be returned if a vector was
    first indexed with a slice and then an integer. The slice must be given in
    standard form.
    """
    start, stop, step = slice_.indices(size)

    if neg_from_end:
        # if the index is negative, the start is the end of the slice, or the
        # end of the array
        # correction if step does not evenly divide into size of array
        # find effective end of slice by counting from start of slice
        end = start + step * math.ceil((stop - start) / step)

        new_index = np.zeros_like(index)
        new_index[index >= 0] = start + index[index >= 0] * step
        new_index[index < 0] = end + index[index < 0] * step
    else:
        new_index = start + index * step

    return new_index


def clean_slice(slice_: slice, size: int) -> slice:
    """Return a slice in standard form, with:
        - start, a positive integer
        - stop, a positive integer or None
        - step, an integer

    We leave the stop as potentially being None, because there is no other way
    to represent at the beginning of the array when the step is negative.
    A slice with stop=-1 has a different interpretation.
    """
    start, stop, step = slice_.indices(size)
    stop = None if stop < 0 else stop
    return slice(start, stop, step)


def init_location(base_shape: Shape) -> StandardLocation:
    # slices in standard form, with step != None
    return [slice(0, size, 1) for size in base_shape]


def shape_from_location(location: StandardLocation, base_shape: Shape) -> Shape:
    """Calculates the expected shape of a numpy array of `base_shape` when
    indexed with `indices`. Assumes that all indices are in standard form, i.e.
    for slices, start/stop are positive integers and step is non-zero integer,
    and that base_shape and indices are the same length.
    """
    # dimension is invisible if indexed with int
    visible_dims = [
        (index, size)
        for index, size in zip(location, base_shape)
        if not isinstance(index, int)
    ]

    # filter out arrays, as they must be handled differently
    arrays = [
        (dim, index)
        for dim, (index, size) in enumerate(visible_dims)
        if isinstance(index, np.ndarray)
    ]
    if arrays:
        broadcasted_shape = np.broadcast(*(arr for _, arr in arrays)).shape
        visible_dims = [
            (index, size)
            for index, size in visible_dims
            if not isinstance(index, np.ndarray)
        ]

    shape = []
    for index, base_size in visible_dims:
        start, stop, step = index.indices(base_size)
        size = max(0, math.ceil((stop - start) / step))
        shape.append(size)

    if arrays:
        first_array = min(dim for dim, _ in arrays)
        shape[first_array:first_array] = list(broadcasted_shape)

    return tuple(shape)


def batch_dims_from_location(location: StandardLocation, n_batch_dims: int) -> int:

    array_indexing = False
    current_batch_dims = n_batch_dims

    for index in location[:n_batch_dims]:
        if isinstance(index, int):
            current_batch_dims -= 1
        elif isinstance(index, np.ndarray):
            if array_indexing:
                current_batch_dims -= 1
            array_indexing = True
    
    return current_batch_dims
