from typing import List, Sequence, Tuple

import numpy as np
from nptyping import NDArray

from parllel.buffers.buffer import Index, Indices


def shift_indices(indices: Indices, shift: int) -> Tuple[Index, ...]:
    if not isinstance(indices, tuple):
        indices = (indices,)
    return shift_index(indices[0], shift) + indices[1:]


def shift_index(index: Index, shift: int) -> Tuple[Index, ...]:
    """Shifts an array index up by an integer value.
    """
    if isinstance(index, int):
        if index < -shift:
            raise IndexError(f"Not enough padding ({shift}) to accomodate "
                             f"index ({index})")
        return (index + shift,)
    if isinstance(index, slice):
        # in case the step is negative, we need to reverse/adjust the limits
        # limits must be incremented because the upper limit of the slice is
        # not in the slice
        # [:] = slice(None, None, None) -> slice(shift, -shift, None)
        # [::-1] = slice(None, None, -1) -> slice(-shift-1, shift-1, -1)
        # [:3:-1] = slice(None, 3, -1) -> slice(-shift-1, 3+shift, -1)
        lower_limit = -(shift+1) if index.step is not None and index.step < 0 else shift
        upper_limit = shift-1 if index.step is not None and index.step < 0 else -shift
        return (slice(
            index.start + shift if index.start is not None else lower_limit,
            index.stop + shift if index.stop is not None else upper_limit,
            index.step,
        ),)
    if index is Ellipsis:
        # add another Ellipsis, to index any remaining dimensions that an
        # Ellipsis would have indexed (possible no extra dimensions remain)
        return (slice(shift, -shift), Ellipsis)
    raise ValueError(index)


def compute_indices(base_array: NDArray, current_array: NDArray):
    """Computes the indices required to index `current_array` from
    `base_array`. For this to work, `current_array` must truly be a subarray of
    `base_array`. Note that indexing a single element of a numpy array results
    in a copy, which is then no longer a subarray of the base array.
    """
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


def does_index_scalar(apparent_shape: Tuple[int, ...], location: Indices):
    """Returns true if an indexing operation at `location` on an array with
    shape `apparent_shape` will index a single element (a scalar), which
    results in a copy for numpy arrays.
    """
    if not isinstance(location, tuple):
        location = (location,)
    return (len(location) == len(apparent_shape) and 
        all(isinstance(index, int) for index in location))


def slicify_final_index(location: Indices):
    """Takes a location which will index a scalar and converts the final index
    into a slice such that an array of shape (1,) is indexed instead. This
    implies that `location` is either an int or a tuple of ints.
    """
    if isinstance(location, tuple):
        return location[:-1] + (slice(location[-1], location[-1] + 1),)
    return slice(location, location + 1)


def add_indices(base_shape: Tuple[int, ...], current_indices: List[Index],
    location: Indices
):
    """Takes an indexing location, the current indices of the subarray relative
    to the base array, and the base array's shape, and returns the next indices
    of the subarray relative to the base array after indexing.

    Issues:
    - no checks to make sure that index is valid (e.g. bounds checking), so the
        indexing has to be done on the ndarray as well.
    - this functions converts slice `s` to standard form (start/stop positive
        integers and step non-zero integer) using `slice(*s.indices(size))`.
        However, does not return a slice in standard form for stop=None and
        step < 0, where the new stop is negative. This results in a "standard"
        slice that gives a different result than the original.
        e.g. slice(5, None, -1) -> slice(5, -1, -1), which is a 0-length slice

    Possible optimizations:
    - move to for loop over location zipped with current_indices with ints
        filtered out
    - it may be possible to jit this function if slices are replaced by tuples
        of 3 integers
    - it may be faster to create a jitclass for Index which represents either
        int, slice, or Ellipsis, and can then be passed to jitted functions
    """
    if isinstance(location, tuple):
        location = list(location)
    else:
        location = [location]

    # check if Ellipsis occurs in location
    i = next((index for index, elem in enumerate(location) if elem is Ellipsis), None)
    if i is not None:
        # pad location with slice(None) elements until length equals ndim
        # assume Ellipsis only occurs once, since the location is well-formed
        location[i:i+1] = [slice(None)] * (len(current_indices) - len(location) + 1)

    i = 0
    for dim, curr_index in enumerate(current_indices):
        if isinstance(curr_index, int):
            # this dimension has already been indexed with an integer, so it is
            # ignored
            continue

        new_index = location[i]
        base_size = base_shape[dim]

        if curr_index == slice(None):
            # this dimension has not yet been indexed at all
            # new index must be cleaned, and then overwrites curr_index
            if new_index == slice(None):
                # don't need to clean this
                pass
            elif isinstance(new_index, int):
                # make negative indices positive
                new_index %= base_size
            else:  # new_index: slice
                # make start/stop positive integers and step non-zero integer
                new_index = slice(*new_index.indices(base_size))
            current_indices[dim] = new_index
        else:
            # this dimension has been indexed with a non-trivial slice
            # add new_index to existing slice
            if isinstance(new_index, int):
                # no need to check bounds on this, we assume the index is
                # correct
                new_index = curr_index.start + new_index * curr_index.step

                current_indices[dim] = new_index
            else:  # new_index: slice
                # get current size of this dimension
                size = shape_from_indices((base_size,), (curr_index,))[0]
                # use the current size to resolve the slice
                start, stop, step = new_index.indices(size)
                current_indices[dim] = slice(
                    curr_index.start + start * curr_index.step,  # start
                    curr_index.start + stop * curr_index.step,   # stop
                    curr_index.step * step,                      # step
                )

        i += 1  # consider next new_index on next loop iteration
        if i == len(location):
            # new_indices exhausted
            break

    return current_indices


def shape_from_indices(base_shape: Tuple[int, ...], indices: Sequence[Index]):
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
