from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Union, Generic

# import numpy as np


Index = Union[int, slice, type(Ellipsis), ]
#TODO: add type(np.newaxis) to Index types?
Indices = Union[Index, Tuple[Index, ...]]

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
