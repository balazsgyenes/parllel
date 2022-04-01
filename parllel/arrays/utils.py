from typing import Dict, Optional, Tuple

import numpy as np
from nptyping import NDArray

from parllel.buffers import (Buffer, NamedArrayTuple, NamedTuple,
    NamedArrayTupleClass_like, dict_to_namedtuple, buffer_map)

from .array import Array


def buffer_from_example(example: Buffer, leading_dims: Tuple[int, ...] = (),
    ArrayClass: Optional[Array] = None, **kwargs) -> Buffer[Array]:
    if example is None:
        return None
    if isinstance(example, (NamedArrayTuple, NamedTuple)):
        buffer_type = NamedArrayTupleClass_like(example)
        return buffer_type(*(buffer_from_example(elem, leading_dims, ArrayClass)
                             for elem in example))
    if isinstance(example, Array):
        shape = leading_dims + example.shape
        dtype = example.dtype
        return type(example)(shape=shape, dtype=dtype, **kwargs)
    else:  # assume np.ndarray
        np_example = np.asarray(example)  # promote scalars to 0d arrays
        shape = leading_dims + np_example.shape
        dtype = np_example.dtype 
        return ArrayClass(shape=shape, dtype=dtype, **kwargs)


def buffer_from_dict_example(example: Dict, leading_dims: Tuple[int, ...], ArrayClass: Array,
                             *, name: str, force_32bit: bool = True, **kwargs) -> Buffer:
    """Create a samples buffer from an example which may be a dictionary (or
    just a single value). The samples buffer will be a NamedArrayTuple with a
    matching structure.
    """
    
    # first, convert dictionary to a namedtuple
    example = dict_to_namedtuple(example, name)

    # convert any Python values to numpy
    example = buffer_map(np.asanyarray, example)

    # demote any 1d scalar arrays to actual scalars
    # this ensures that the final buffer with leading dimensions is the right size
    def to_numpy_scalar(arr):
        if arr.shape == (1,):
            return arr[0]
        return arr
    example = buffer_map(to_numpy_scalar, example)

    # force float64 arrays to float32 arrays to save memory
    if force_32bit:
        def force_float_int_to_32bit(arr: NDArray):
            if arr.dtype == np.float64:
                return arr.astype(np.float32)
            elif arr.dtype == np.int64:
                return arr.astype(np.int32)
            return arr

        example = buffer_map(force_float_int_to_32bit, example)

    return buffer_from_example(example, leading_dims, ArrayClass, **kwargs)
