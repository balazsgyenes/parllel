from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar, Union


# A single index element, e.g. arr[3:6]
Index = Union[int, slice, type(Ellipsis)]

# A single indexing location, e.g. arr[2, 0] or arr[:-2]
Indices = Union[Index, Tuple[Index, ...]]

# Represents the object type at the leaves of the buffer tree structure:
# e.g. numpy array, Array, torch tensor, jax array, etc.
LeafType = TypeVar('LeafType')


class Buffer(ABC, Generic[LeafType]):
    """A buffer represents a tree-like structure, where the non-leaf nodes are
    either NamedArrayTuples, and the leaf nodes are array-like data containers,
    such as Arrays, numpy ndarrays, torch tensors, etc. For convenience, many
    buffer utility methods also support tuples and NamedTuples of leaf nodes,
    but these are not true Buffers.
    """
    @property
    def index_history(self) -> Tuple[Indices, ...]:
        # return a copy of the list as a tuple, not the list itself
        return tuple(self._index_history)

    @property
    def buffer_id(self) -> int:
        return self._buffer_id

    @abstractmethod
    def __getitem__(self, location: Indices) -> Union[Buffer[LeafType], LeafType]:
        pass
