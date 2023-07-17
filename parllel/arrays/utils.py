from typing import Literal, Union

import numpy as np

from parllel.buffers import (
    Buffer,
    NamedArrayTuple,
    NamedArrayTupleClass_like,
    NamedTuple,
    dict_to_namedtuple,
)

from .array import Array


def buffer_from_example(
    example: Buffer,
    *,
    force_32bit: Literal[True, "float", "int", False] = True,
    inherit_full_size: bool = False,
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
                    force_32bit=force_32bit,
                    inherit_full_size=inherit_full_size,
                    **kwargs,
                )
                for elem in example
            )
        )
    if isinstance(example, Array):
        return example.new_array(inherit_full_size=inherit_full_size, **kwargs)
    else:
        return Array.from_numpy(example, force_32bit=force_32bit, **kwargs)


def buffer_from_dict_example(
    example: Union[np.ndarray, dict],
    *,
    name: str,
    force_32bit: Literal[True, "float", "int", False] = True,
    inherit_full_size: bool = False,
    **kwargs,
) -> Buffer[Array]:
    """Create a samples buffer from an example which may be a dictionary (or
    just a single value). The samples buffer will be a NamedArrayTuple with a
    matching structure.
    """

    # first, convert dictionary to a namedtuple
    example = dict_to_namedtuple(example, name)

    return buffer_from_example(
        example,
        force_32bit=force_32bit,
        inherit_full_size=inherit_full_size,
        **kwargs,
    )
