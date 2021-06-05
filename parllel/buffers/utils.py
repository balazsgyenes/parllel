from typing import Tuple

import numpy as np

from parllel.buffers import Buffer
from parllel.arrays import Array
from .named_tuple import NamedArrayTuple, NamedTuple, NamedArrayTupleClass_like

def buffer_from_example(example, leading_dims: Tuple[int, ...], ArrayClass: Array, **kwargs) -> Buffer:
    if example is None:
        return None
    if isinstance(example, NamedArrayTuple, NamedTuple):
        buffer_type = NamedArrayTupleClass_like(example)
        return buffer_type(*(buffer_from_example(elem, leading_dims,
            ArrayClass=ArrayClass)
            for elem in example))
    else:
        np_example = np.asarray(example)
        shape = leading_dims + np_example.shape
        dtype = np_example.dtype 
        return ArrayClass(shape=shape, dtype=dtype, **kwargs)
