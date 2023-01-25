from typing import Dict, Tuple, Type

import numpy as np

from parllel.buffers import (Buffer, NamedArrayTuple, NamedTuple,
    NamedArrayTupleClass_like, dict_to_namedtuple)

from .array import Array


def buffer_from_example(
    example: Buffer,
    leading_dims: Tuple[int, ...],
    ArrayClass: Type[Array],
    *,
    force_32bit: bool = True,
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
                    ArrayClass,
                    force_32bit=force_32bit,
                    **kwargs,
                )
                for elem in example
            )
        )
    if isinstance(example, Array):
        shape = example.shape
        shape = kwargs.pop("shape", None) or shape  # TODO: do not pop (dict is shared)
        shape = leading_dims + shape
        # TODO: ensure compatibility for public Array subclasses
        return ArrayClass.like(example, shape=shape, **kwargs)
    else:
        np_example = np.asanyarray(example)  # promote scalars to 0d arrays
        shape = leading_dims + np_example.shape
        dtype = np_example.dtype 
        if force_32bit:
            # TODO: add more options here, e.g. False, "int", "float", and True
            if dtype == np.int64:
                dtype = np.int32
            elif dtype == np.float64:
                dtype = np.float32
        return ArrayClass(shape=shape, dtype=dtype, **kwargs)


def buffer_from_dict_example(
    example: Dict,
    leading_dims: Tuple[int, ...],
    ArrayClass: Type[Array],
    *,
    name: str,
    force_32bit: bool = True,
    **kwargs,
) -> Buffer[Array]:
    """Create a samples buffer from an example which may be a dictionary (or
    just a single value). The samples buffer will be a NamedArrayTuple with a
    matching structure.
    """
    
    # first, convert dictionary to a namedtuple
    example = dict_to_namedtuple(example, name)

    return buffer_from_example(example, leading_dims, ArrayClass, force_32bit, **kwargs)
