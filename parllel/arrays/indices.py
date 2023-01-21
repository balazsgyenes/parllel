from typing import List, Sequence, Tuple

import numpy as np
from nptyping import NDArray

from parllel.buffers.buffer import Index, Indices


def compute_indices(base_array: NDArray, current_array: NDArray):
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


def predict_copy_on_index(apparent_shape: Tuple[int, ...], location: Indices):
    return (len(location) == len(apparent_shape) and 
        all(isinstance(index, int) for index in location))


def add_indices(current_indices: List[Index], location: Indices):
    """Takes an indexing location, the current indices of the subarray relative
    to the base array, and the base array's shape, and returns the next indices
    of the subarray relative to the base array after indexing.

    Issues:
    - no checks to make sure that index is valid (e.g. bounds checking), so the
        indexing has to be done on the ndarray as well.

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
            # this dimension has already been indexed with an integer, it
            # cannot be indexed further
            continue

        new_index = location[i]

        if curr_index == slice(None):
            # this dimension has not yet been indexed at all
            # new_index must be cleaned, and then overwrites curr_index
            if isinstance(new_index, slice):
                step = new_index.step or 1
                new_index = slice(
                    new_index.start or (0 if step > 0 else -1),
                    new_index.stop,
                    step,
                )
            current_indices[dim] = new_index
        else:
            # this dimension has been indexed with a non-trivial slice
            # add new_index to existing slice
            if isinstance(new_index, int):
                current_indices[dim] = index_slice(curr_index, new_index)

            # no op if indexed with the trivial slice
            elif new_index != slice(None):  # new_index: slice

                step = curr_index.step
                # translate new_index.start into "world coordinates"
                start = index_slice(curr_index, new_index.start or (
                    0 if step > 0 else -1
                ))
                
                if (new_stop := new_index.stop) is not None:
                    # translate new_stop into "world coordinates"
                    new_stop = index_slice(curr_index, new_stop)

                    stop = (
                        min(curr_stop, new_stop)
                        if (curr_stop := curr_index.stop) is not None else
                        new_stop
                    )
                else:
                    # curr_index.stop might be None, that's okay
                    stop = curr_index.stop

                # update step by multiplying new step onto it
                if new_step := new_index.step is not None:
                    step *= new_step

                current_indices[dim] = slice(start, stop, step)

        i += 1  # consider next new_index on next loop iteration
        if i == len(location):
            # new_indices exhausted
            break

    return current_indices


def index_slice(current_slice: slice, new_index: int) -> int:
    step = current_slice.step
    start = (
        current_slice.start
        if new_index >= 0 else
        # if index negative, determine index of the end of the array
        current_slice.stop or (
            0  # end=None means we must index from just past the -1th element
            if step > 0 else  # end of flipped array is the beginning
            -1  # end=None means we must index from just before the 0th element
        )
    )
    return start + new_index * step


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
