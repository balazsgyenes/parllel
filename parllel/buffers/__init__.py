from .buffer import Buffer, Index, Indices
from .named_tuple import (
    NamedTuple, NamedTupleClass,
    NamedArrayTuple, NamedArrayTupleClass,
)

__all__ = [
    Buffer, Index, Indices,
    NamedTuple, NamedTupleClass,
    NamedArrayTuple, NamedArrayTupleClass,
]

"""
TODO: most array functions need to be defined at the buffer level, so calls like
observation.rotate() do not fail if observation happens to be a namedarraytuple
instead of a single Array
"""
