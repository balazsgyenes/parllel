from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, Tuple, TypeVar, Union


# A single index element, e.g. arr[3:6]
Index = Union[int, slice, type(Ellipsis)]

# A single indexing location, e.g. arr[2, 0] or arr[:-2]
Indices = Union[Index, Tuple[Index, ...]]

# Represents the object type at the leaves of the buffer tree structure:
# e.g. numpy array, Array, torch tensor, jax array, etc.
LeafType = TypeVar('LeafType')


class Buffer(ABC, Generic[LeafType]):
    """A buffer represents a tree-like structure, where the non-leaf nodes are
    either tuples, NamedTuples, or NamedArrayTuples, and the leaf nodes are
    Array objects, numpy arrays, torch tensors, etc.
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


class VoidBuffer(Buffer):
    def __init__(self) -> None:
        self._buffer_id = id(self)
        self._index_history = []

    def __getitem__(self, location: Indices) -> Union[Buffer[LeafType], LeafType]:
        return self

    def __setitem__(self, location: Indices, value: Any) -> None:
        pass
