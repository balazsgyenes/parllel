from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Union

# import numpy as np


Index = Union[int, slice, type(Ellipsis), ]
#TODO: add type(np.newaxis) to Index types?
Indices = Union[Index, Tuple[Index, ...]]


class Buffer(ABC):
    @property
    def index_history(self) -> Tuple[Tuple[Index, ...], ...]:
        # return a copy of the list as a tuple, not the list itself
        return tuple(self._index_history)

    @property
    def buffer_id(self) -> int:
        return self._buffer_id

    @abstractmethod
    def __getitem__(self, location: Indices) -> Buffer:
        pass