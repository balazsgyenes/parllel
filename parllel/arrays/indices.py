import math
from typing import Sequence, TypeVar

import numpy as np

from parllel.buffers.buffer import Index, Indices


def compute_indices(base_array: np.ndarray, current_array: np.ndarray):
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

    for (curr_size, curr_stride) in zip(curr_shape, curr_strides):
        # search for the corresponding stride in base_array
        # base_stride_(dim-1) > abs(current_stride_(dim)) >= base_stride_(dim)
        # the absolute value is required because the current stride might be negative
        dim = next(index for index, elem in enumerate(base_strides) if abs(curr_stride) >= elem)
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
    
    return tuple(current_indices)


def predict_copy_on_index(shape: tuple[int, ...], new_location: Indices):
    return (len(new_location) == len(shape) and 
        all(isinstance(index, int) for index in new_location))


def add_locations(
    current_location: Sequence[Index],
    new_location: Indices,
    base_shape: tuple[int, ...],
) -> list[Index]:
    """Takes an indexing location, the current indices of the subarray relative
    to the base array, and the base array's shape, and returns the next indices
    of the subarray relative to the base array after indexing.

    If current_location contains slices, they must be in standard form (see
    `clean_slice`).
    """
    if any(isinstance(loc, np.ndarray) for loc in current_location):
        raise IndexError(
            "Cannot processing further indexing operations after advanced "
            "indexing."
        )

    current_location = list(current_location)  # create a copy to prevent modifying inputs

    if isinstance(new_location, tuple):
        new_location = list(new_location)
    else:
        new_location = [new_location]

    # check if Ellipsis occurs in new_location
    i = next((index for index, elem in enumerate(new_location) if elem is Ellipsis), None)
    if i is not None:
        # pad new_location with slice(None) elements until length equals the
        # apparent number of dimensions
        # assume Ellipsis only occurs once, since the location is well-formed
        apparent_n_dim = len([loc for loc in current_location if isinstance(loc, slice)])
        new_location[i:i+1] = [slice(None)] * (apparent_n_dim - len(new_location) + 1)

    if not new_location:
        # e.g. if new_location was [...] or [()]
        return current_location

    i = 0
    for dim, (current_index, size) in enumerate(zip(current_location, base_shape)):
        if isinstance(current_index, int):
            # this dimension has already been indexed with an integer, it
            # cannot be indexed further
            continue

        current_location[dim] = add_indices(current_index, new_location[i], size)

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

    return current_location


IndexType = TypeVar("IndexType", int, slice, np.ndarray)
def add_indices(current_index: slice, new_index: IndexType, size: int) -> IndexType:
    """Takes the current indexing state of a single dimension and a new index,
    and returns a single index (int or slice) that could be used to index the
    base array to achieve the same result.

    current_index must be a slice in standard form (see `clean_slice`).
    """
    if isinstance(new_index, np.ndarray):
        index = np_index_slice(current_index, new_index, size)
        if not np.all((0 <= index) & (index < size)):
            out_of_bounds = new_index[~(0 <= index)]
            if out_of_bounds.shape[0] == 0:
                out_of_bounds = new_index[~(index < size)]
            raise IndexError(
                f"Index {out_of_bounds[0]} is out of bounds for axis with size "
                f"{size}."
            )
        return index

    elif isinstance(new_index, int):
        index = index_slice(current_index, new_index, size)
        if not (0 <= index < size):
            raise IndexError(
                f"Index {new_index} is out of bounds for axis with size "
                f"{size}."
            )
        return index  # index should never be negative after cleaning

    elif new_index == slice(None):
        # no op if indexed with the trivial slice
        return current_index
    
    else:  # new_index: slice
        # convert to numerical form for computation
        start, stop, step = current_index.indices(size)

        new_step = new_index.step if new_index.step is not None else 1
        new_start = new_index.start if new_index.start is not None else (
            0 if new_step > 0 else -1
        )
        new_stop = new_index.stop

        # translate new_index.start into "world coordinates"
        new_start = index_slice(current_index, new_start, size)
        new_start = max(0, new_start)  # lower bound at 0
        
        if new_stop is not None:
            # translate new_stop into "world coordinates"
            new_stop = index_slice(current_index, new_stop, size)
            # find the stop that results in the shorter sequence
            new_stop = min(stop, new_stop) if step > 0 else max(stop, new_stop)
            # bound at the relevant end of the array, preserving empty slices
            new_stop = max(0, new_stop) if step > 0 else min(new_stop, size)
        elif new_step > 0:
            new_stop = stop
        else:
            new_stop = start
            # extend new_stop one unit away from body of range
            new_stop += np.sign(step * new_step)
        
        # set to None instead of -1
        # this might happen if the stop was specified as an integer, but lands
        # before the beginning of the array, or if the stop was None and the
        # start/stop of current_index was -1 (i.e. None with negative step)
        new_stop = None if new_stop < 0 else new_stop

        # update step by multiplying new step onto it
        new_step = step * new_step
        
        return slice(new_start, new_stop, new_step)


def index_slice(slice_: slice, index: int, size: int) -> int:
    """Compute the index of the element that would be returned if a vector was
    first indexed with a slice and then an integer. The slice must be given in
    standard form.
    """
    step = slice_.step
    if index >= 0:
        zero = slice_.start
    else:
        # if the index is negative, the start is the end of the slice, or the
        # end of the array
        start, stop, step = slice_.indices(size)

        # correction if step does not evenly divide into size of array
        # find effective end of slice by counting from start of slice
        zero = start + step * math.ceil((stop - start) / step)

    return zero + index * step


def np_index_slice(slice_: slice, index: np.ndarray, size: int) -> int:
    """Compute the index of the element that would be returned if a vector was
    first indexed with a slice and then an integer. The slice must be given in
    standard form.
    """
    start, stop, step = slice_.indices(size)
    # if the index is negative, the start is the end of the slice, or the
    # end of the array
    # correction if step does not evenly divide into size of array
    # find effective end of slice by counting from start of slice
    end = start + step * math.ceil((stop - start) / step)

    new_index = np.zeros_like(index)
    new_index[index >= 0] = start + index[index >= 0] * step
    new_index[index < 0] = end + index[index < 0] * step

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


def init_location(base_shape: tuple[int, ...]) -> list[Index]:
    # slices in standard form, with step != None
    return [slice(0, size, 1) for size in base_shape]


def shape_from_indices(base_shape: tuple[int, ...], indices: Sequence[Index]):
    """Calculates the expected shape of a numpy array of `base_shape` when
    indexed with `indices`. Assumes that all indices are in standard form, i.e.
    for slices, start/stop are positive integers and step is non-zero integer,
    and that base_shape and indices are the same length.
    """
    return tuple(
        (
            size # dimension unindexed, base size unchanged
            if index == slice(None)
            # otherwise calculate size of slice
            else max(
                # (stop - start) // step, but also account for the fact that
                # stop is not included in slice. increment depending on whether
                # step is negative or positive
                (index.stop - np.sign(index.step) - index.start) // index.step + 1,
                0) # in case stop is before start, clamp size to at least 0
        )
        for size, index
        in zip(base_shape, indices)
        if not isinstance(index, int)  # dimension is invisible if indexed with int
    )
