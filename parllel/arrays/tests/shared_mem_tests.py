import multiprocessing as mp
import ctypes

import numpy as np


shape = (5,5)
dtype = np.int32

size = int(np.prod(shape))
nbytes = size * np.dtype(dtype).itemsize

def f(obj):
    # buffer = pipe.recv()
    buffer = obj
    array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
    array.fill(5)

if __name__ == "__main__":
    mp.set_start_method("spawn")

    raw_array = mp.RawArray(ctypes.c_char, nbytes)
    array = np.frombuffer(raw_array, dtype=dtype, count=size).reshape(shape)
    # array = np.array(raw_array, dtype, copy=False)
    print(array)

    # parent_pipe, child_pipe = mp.Pipe()

    p = mp.Process(target=f, args=(raw_array,))
    p.start()
    # parent_pipe.send(raw_array)
    p.join()

    print(array)
    print(mp.get_start_method())