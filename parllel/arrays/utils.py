from typing import Dict, Literal, Tuple

import numpy as np

from parllel.buffers import (Buffer, NamedArrayTuple, NamedTuple,
    NamedArrayTupleClass_like, dict_to_namedtuple)

from .array import Array


def buffer_from_example(
    example: Buffer,
    leading_dims: Tuple[int, ...] = (),
    *,
    force_32bit: Literal[True, "float", "int", False] = True,
    **kwargs,
) -> Buffer[Array]:
    if example is None:
        return None
    if isinstance(example, (NamedArrayTuple, NamedTuple)):
        buffer_type = NamedArrayTupleClass_like(example)
        return buffer_type(
            *(
                buffer_from_example(
                    elem,
                    leading_dims,
                    force_32bit=force_32bit,
                    **kwargs,
                )
                for elem in example
            )
        )
    if isinstance(example, Array):
        shape = kwargs.pop("shape", None) or example.shape
        shape = leading_dims + shape
        return Array.like(example, shape=shape, **kwargs)
    else:
        np_example = np.asanyarray(example)  # promote scalars to 0d arrays
        shape = kwargs.pop("shape", None) or np_example.shape
        shape = leading_dims + shape
        if (dtype := kwargs.pop("dtype", None)) is None:
            dtype = np_example.dtype
            if dtype == np.int64 and force_32bit in {True, "int"}:
                dtype = np.int32
            elif dtype == np.float64 and force_32bit in {True, "float"}:
                dtype = np.float32
        return Array(shape=shape, dtype=dtype, **kwargs)


def buffer_from_dict_example(
    example: Dict,
    leading_dims: Tuple[int, ...] = (),
    *,
    name: str,
    force_32bit: Literal[True, "float", "int", False] = True,
    **kwargs,
) -> Buffer[Array]:
    """Create a samples buffer from an example which may be a dictionary (or
    just a single value). The samples buffer will be a NamedArrayTuple with a
    matching structure.
    """
    
    # first, convert dictionary to a namedtuple
    example = dict_to_namedtuple(example, name)

    return buffer_from_example(example, leading_dims, force_32bit=force_32bit, **kwargs)
