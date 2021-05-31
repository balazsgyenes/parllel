from functools import reduce
import multiprocessing as mp
from parllel.arrays.sharedmemory import SharedMemoryArray
from typing import Dict

from parllel.buffers import Buffer, NamedArrayTuple
from parllel.arrays import SharedMemoryArray, RotatingSharedMemoryArray


_buffer_registry: Dict[int, Buffer] = {}


def rebuild_buffer(buffer_id, index_history):
    # print(f"Rebuilding unregistered buffer {buffer_id}")
    try:
        base = _buffer_registry[buffer_id]
    except KeyError as e:
        raise RuntimeError(f"Cannot unpickle unregistered buffer with id '{buffer_id}.") from e

    # apply each index in index_history to base in succession, and return result
    return reduce(lambda buf, index: buf[index], index_history, base)


def reduce_buffer(buffer: Buffer):
    try:
        _ = _buffer_registry[buffer.buffer_id]
        # print(f"Reducing registered buffer {buffer.buffer_id}")
        return rebuild_buffer, (buffer.buffer_id, buffer.index_history)
    except KeyError as e:
        raise RuntimeError(f"Cannot pickle unregistered buffer with id '{buffer.buffer_id}.") from e


def register_shared_memory_buffer(buffer: Buffer):
    """Register a buffer to be pickled by handle, not by value. For
    compatibility with both fork and spawn, call in each process after process
    start.
    """
    # print(f"Registering buffer {buffer.buffer_id}.")
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
        ctx.reducer.register(t, reduce_buffer)
