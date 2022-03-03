from typing import Dict, Tuple

import numpy as np

from parllel.buffers import Buffer
from parllel.arrays import Array
from .named_tuple import NamedArrayTuple, NamedTuple, NamedArrayTupleClass_like, buffer_func, dict_to_namedtuple


def buffer_from_example(example, leading_dims: Tuple[int, ...], ArrayClass: Array, **kwargs) -> Buffer:
    if example is None:
        return None
    if isinstance(example, (NamedArrayTuple, NamedTuple)):
        buffer_type = NamedArrayTupleClass_like(example)
        return buffer_type(*(buffer_from_example(elem, leading_dims, ArrayClass)
                             for elem in example))
    else:
        np_example = np.asarray(example)  # promote scalars to 0d arrays
        shape = leading_dims + np_example.shape
        dtype = np_example.dtype 
        return ArrayClass(shape=shape, dtype=dtype, **kwargs)


def buffer_from_dict_example(example, leading_dims: Tuple[int, ...], ArrayClass: Array,
                             *, name: str, force_float32: bool = False, **kwargs) -> Buffer:
    """Create a samples buffer from an example which may be a dictionary (or
    just a single value). The samples buffer will be a NamedArrayTuple with a
    matching structure.
    """
    
    # first, convert dictionary to a namedtuple
    example = dict_to_namedtuple(example, name)

    # convert any Python values to numpy
    example = buffer_func(np.asanyarray, example)

    # demote any 1d scalar arrays to actual scalars
    # this ensures that the final buffer with leading dimensions is the right size
    def to_numpy_scalar(arr):
        if arr.shape == (1,):
            return arr[0]
        return arr
    example = buffer_func(to_numpy_scalar, example)

    # force float64 arrays to float32 arrays to save memory
    if force_float32:
        def force_float_to_float32(arr):
            if arr.dtype == np.float64:
                return arr.astype(np.float32)
            return arr

        example = buffer_func(force_float_to_float32, example)

    return buffer_from_example(example, leading_dims, ArrayClass, **kwargs)