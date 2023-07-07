from typing import Any, Union

from .buffer import Buffer, Indices, LeafType


class VoidBuffer(Buffer):
    """A buffer type that acts as a data sink. Data written into this buffer
    is simply lost.
    """
    def __init__(self) -> None:
        self._buffer_id = id(self)
        self._index_history = []

    def __getitem__(self, location: Indices) -> Union[Buffer[LeafType], LeafType]:
        return self

    def __setitem__(self, location: Indices, value: Any) -> None:
        pass

    def close(self) -> None:
        pass
