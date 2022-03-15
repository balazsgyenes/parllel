import multiprocessing as mp
from typing import Sequence
import pytest

import numpy as np

# from parllel.buffers.named_tuple import (
#     NamedTuple, NamedTupleClass,
#     NamedArrayTuple, NamedArrayTupleClass,
#     NamedArrayTupleBuffer, NamedArrayTupleBufferClass,
# )
from parllel.arrays import SharedMemoryArray
from parllel.buffers import Buffer
import parllel.buffers.reductions as reductions

arr = SharedMemoryArray(shape=(5,4,3), dtype=np.int32)
arr.initialize()
arr[:] = np.arange(np.prod(arr.shape)).reshape(arr.shape)
arr2 = SharedMemoryArray(shape=(6,2), dtype=np.int32)
arr2.initialize()
arr3 = SharedMemoryArray(shape=(8,3), dtype=np.int32)
arr3.initialize()
buffers = (arr2, (arr, arr3))

rebuild = reductions.rebuild_buffer
def debug_rebuild_buffer(buffer_id, index_history):
    print(f"Rebuilding unregistered buffer {buffer_id}")
    return rebuild(buffer_id, index_history)
reductions.rebuild_buffer = debug_rebuild_buffer

reduce = reductions.reduce_buffer
def debug_reduce_buffer(buffer):
    try:
        print(f"Reducing registered buffer {buffer.buffer_id}")
    except AttributeError:
        # error will be handled by reduce_buffer
        pass
    return reduce(buffer)
reductions.reduce_buffer = debug_reduce_buffer

register = reductions.register_shared_memory_buffer
def debug_register_shared_memory_buffer(buffer):
    print(f"Registering buffer {buffer.buffer_id}.")
    return register(buffer)
reductions.register_shared_memory_buffer = debug_register_shared_memory_buffer


def test_buffer_piping():
    mp_context = mp.get_context("spawn")
    parent_pipe, child_pipe = mp_context.Pipe()
    buffers = (arr, arr2, arr3)
    for buffer in buffers:
        reductions.register_shared_memory_buffer(buffer)
    p = mp_context.Process(target=recv_buffer_piping, args=(buffers, child_pipe))
    p.start()
    reductions.register_buffer_pickler(mp_context)
    parent_pipe.send(arr)
    p.join()

def recv_buffer_piping(buffers: Sequence[Buffer], child_pipe: mp.Pipe):
    # registering is necessary is starting method is spawn
    for buffer in buffers:
        reductions.register_shared_memory_buffer(buffer)
    recv_arr = child_pipe.recv()
    assert np.array_equal(recv_arr, arr)

def test_buffer_view_piping():
    mp_context = mp.get_context("spawn")
    parent_pipe, child_pipe = mp_context.Pipe()
    buffers = (arr, arr2, arr3)
    for buffer in buffers:
        reductions.register_shared_memory_buffer(buffer)
    p = mp_context.Process(target=recv_buffer_view_piping, args=(buffers, child_pipe))
    p.start()
    reductions.register_buffer_pickler(mp_context)
    parent_pipe.send(arr[2])
    p.join()

def recv_buffer_view_piping(buffers: Sequence[Buffer], child_pipe: mp.Pipe):
    # registering is necessary is starting method is spawn
    for buffer in buffers:
        reductions.register_shared_memory_buffer(buffer)
    recv_arr = child_pipe.recv()
    assert np.array_equal(recv_arr, arr[2])

if __name__ == "__main__":
    """For some reason, only one test can be run at a time.
    """
    # test_buffer_piping()
    test_buffer_view_piping()