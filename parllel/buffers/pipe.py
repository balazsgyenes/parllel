from dataclasses import dataclass
from functools import reduce
import io
import pickle
from multiprocessing.connection import Connection
from typing import Any, Dict, Tuple

from parllel.buffers import Buffer, Indices
from parllel.arrays import SharedMemoryArray, ManagedMemoryArray
from .pickler import BufferPickler, BufferUnpickler

class BufferPipe:
    def __init__(self, pipe: Connection) -> None:
        self._pipe = pipe
        self._buffer_registry: Dict[int, Buffer] = {}

    def send(self, obj: Any) -> None:
        buf = io.BytesIO()
        BufferPickler(buf, pickle.HIGHEST_PROTOCOL, buffer_registry=self._buffer_registry).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self) -> Any:
        obj = self.recv_bytes()
        buf = io.BytesIO(obj)
        return BufferUnpickler(buf, buffer_registry=self._buffer_registry).load()

    def register_buffer(self, buffer: Buffer):
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
