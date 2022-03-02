from dataclasses import dataclass
from functools import reduce
from multiprocessing.connection import Connection
from typing import Any, Dict, Tuple

from parllel.buffers import Buffer, Indices
from parllel.arrays import SharedMemoryArray, ManagedMemoryArray


@dataclass
class BufferReference:
    buffer_id: int
    index_history: Tuple[Indices]


class BufferPipe:
    def __init__(self, pipe: Connection) -> None:
        self._pipe = pipe
        self._buffer_registry: Dict[int, Buffer] = {}

    def send(self, obj: Any) -> None:
        if isinstance(obj, Buffer):
            try:
                _ = self._buffer_registry[obj.buffer_id]
                ref = BufferReference(obj.buffer_id, obj.index_history)
                return self._pipe.send(ref)
            except KeyError:
                # for unregistered buffers, fallback to normal pickling, but
                # register the buffer to avoid pickling next time
                self.register_buffer(obj)
        self._pipe.send(obj)

    def recv(self) -> Any:
        obj = self._pipe.recv()
        if isinstance(obj, BufferReference):
            try:
                base = self._buffer_registry[obj.buffer_id]
            except KeyError as e:
                raise RuntimeError("Received reference to unregistered buffer"
                    f" with id: {obj.buffer_id}.") from e

            # apply each index in index_history to base in succession, and
            # return result
            return reduce(lambda buf, index: buf[index], obj.index_history, base)
        
        elif isinstance(obj, Buffer):
            # buffer has already been registered on the sender side, so register
            # here as well
            self.register_buffer(obj)

        return obj

    def register_buffer(self, buffer: Buffer):
        """This method should only be called before process start."""
        self._buffer_registry[buffer.buffer_id] = buffer
        if isinstance(buffer, tuple):
            for element in buffer:
                self.register_buffer(element)
        else:
            assert isinstance(buffer, (SharedMemoryArray, ManagedMemoryArray)), (
                "Only arrays in shared memory (or managed shared memory) can "
                "be moved between processes.")

    def __getattr__(self, name: str) -> Any:
        if '_pipe' in self.__dict__:
            return getattr(self._pipe, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, '_pipe'))
