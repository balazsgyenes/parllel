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


def add_locations(current_location: list[Index], new_location: Indices):
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
    if isinstance(new_location, tuple):
        new_location = list(new_location)
    else:
        new_location = [new_location]

    # check if Ellipsis occurs in new_location
    i = next((index for index, elem in enumerate(new_location) if elem is Ellipsis), None)
    if i is not None:
        # pad new_location with slice(None) elements until length equals ndim
        # assume Ellipsis only occurs once, since the location is well-formed
        new_location[i:i+1] = [slice(None)] * (len(current_location) - len(new_location) + 1)

    i = 0
    for dim, current_index in enumerate(current_location):
        if isinstance(current_index, int):
            # this dimension has already been indexed with an integer, it
            # cannot be indexed further
            continue

        current_location[dim] = add_indices(current_index, new_location[i])

        i += 1  # consider next new_index on next loop iteration
        if i == len(new_location):
            # new_location exhausted
            break
    else:
        # if new_location has not been exhausted, just add the remaining
        # elements to current_indices after cleaning them
        current_location.extend(
            add_indices(slice(None), new_index)
            for new_index in new_location[i:]
        )

    return current_location


IndexType = TypeVar("IndexType", int, slice)
def add_indices(current_index: slice, new_index: IndexType) -> IndexType:
    if current_index == slice(None):
        # this dimension has not yet been indexed at all
        # new_index must be cleaned, and then overwrites current_index
        if isinstance(new_index, slice):
            step = new_index.step or 1
            if (start := new_index.start) is None:
                start = 0 if step > 0 else -1
            # we leave the stop as potentially being None because:
            # 1. we do not know the size of this dimension
            # 2. there is no way to represent an endpoint of None with a
            # negative step, even if we know the size
            new_index = slice(start, new_index.stop, step)
        return new_index
    else:
        # this dimension has been indexed with a non-trivial slice
        # add new_index to existing slice
        if isinstance(new_index, int):
            return index_slice(current_index, new_index)

        # no op if indexed with the trivial slice
        elif new_index != slice(None):  # new_index: slice

            step = current_index.step
            # translate new_index.start into "world coordinates"
            start = index_slice(
                current_index,
                new_index.start if new_index.start is not None else (
                    0 if step > 0 else -1
                ),
            )
            
            if (new_stop := new_index.stop) is not None:
                # translate new_stop into "world coordinates"
                new_stop = index_slice(current_index, new_stop)

                stop = (
                    (
                        min(curr_stop, new_stop)
                        if step > 0 else
                        max(curr_stop, new_stop)
                    )
                    if (curr_stop := current_index.stop) is not None else
                    new_stop
                )
            else:
                # current_index.stop might be None, that's okay
                # TODO: needs to be flipped if new_step is negative
                stop = current_index.stop

            # update step by multiplying new step onto it
            if (new_step := new_index.step) is not None:
                step *= new_step

            return slice(start, stop, step)

        return current_index


def index_slice(current_slice: slice, index: int) -> int:
    step = current_slice.step
    if index >= 0:
        start = current_slice.start
    else:
        # if index negative, determine index of the end of the array
        if (start := current_slice.stop) is None:
            start = (
                -1 + step  # end=None means we must index from just past the -1th element
                if step > 0 else  # end of flipped array is the beginning
                0 + step  # end=None means we must index from just before the 0th element
            )
        else:
            # corrects for case when step does not divide the distance between
            # start and stop of slice
            if (rem := (start - current_slice.start) % step) > 0:
                # round up to nearest multiple of step
                start += (step - rem)
    return start + index * step


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
