import multiprocessing as mp
import ctypes

import numpy as np

from parllel.arrays.sharedmemory import SharedMemoryArray

shape = (5,5)
dtype = np.int32

size = int(np.prod(shape))
nbytes = size * np.dtype(dtype).itemsize

def f(obj):
    # buffer = obj
    # array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
    array = obj
    array[:, :] = 5

if __name__ == "__main__":
    mp.set_start_method("spawn")

    # raw_array = mp.RawArray(ctypes.c_char, nbytes)
    # array = np.frombuffer(raw_array, dtype=dtype, count=size).reshape(shape)
    array = SharedMemoryArray(shape=shape, dtype=dtype)
    print(array)
    array.initialize()
    print(array)


    p = mp.Process(target=f, args=(array,))
    p.start()
    p.join()

    print(array)
    print(mp.get_start_method())