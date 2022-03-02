from functools import reduce
import multiprocessing as mp
from parllel.arrays.sharedmemory import SharedMemoryArray
from typing import Dict

from parllel.buffers import Buffer, NamedArrayTuple
from parllel.arrays import SharedMemoryArray, RotatingSharedMemoryArray


"""Process-global registry of buffers. Any registered buffers can be sent quickly
between processes by buffer_id.
"""
_buffer_registry: Dict[int, Buffer] = {}


def rebuild_buffer(buffer_id, index_history):
    try:
        base = _buffer_registry[buffer_id]
    except KeyError as e:
        raise RuntimeError(f"Cannot unpickle unregistered buffer with id '{buffer_id}.") from e

    # apply each index in index_history to base in succession, and return result
    return reduce(lambda buf, index: buf[index], index_history, base)


def reduce_buffer(buffer: Buffer):
    try:
        _ = _buffer_registry[buffer.buffer_id]
        return rebuild_buffer, (buffer.buffer_id, buffer.index_history)
    except KeyError:
        # for unregistered buffers, fall back to default behaviour. this might
        # be expensive (or even silly), but it should not happen often
        if isinstance(buffer, NamedArrayTuple):
            return NamedArrayTuple.__new__, (NamedArrayTuple, *buffer.__getnewargs__())
        return buffer.__reduce__()


def register_shared_memory_buffer(buffer: Buffer):
    """Register a buffer to be pickled by handle, not by value. For
    compatibility with both fork and spawn, call in each process after process
    start.
    """
    _buffer_registry[buffer.buffer_id] = buffer
    if isinstance(buffer, tuple):
        for element in buffer:
            register_shared_memory_buffer(element)


def register_buffer_pickler(ctx = None):
    """Call this method after process start.
    """
    if ctx is None:
        ctx = mp.get_context()
    for t in (SharedMemoryArray, RotatingSharedMemoryArray, NamedArrayTuple):
        # not needed for ManagedMemoryArray because it can always be pickled by
        # looking it up by name
        ctx.reducer.register(t, reduce_buffer)
