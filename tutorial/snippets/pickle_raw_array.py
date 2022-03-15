"""This snippet demonstrates that multiprocessing.RawArray can be passed to a
forked process on startup, whereas a numpy array in shared memory cannot.
However, the RawArray cannot be passed in a Pipe (or a Queue).
"""

import multiprocessing as mp
import numpy as np
import ctypes


shape = (5,5)

def main():
    # when creating processes using spawn, resources must be pickled to the
    # child process after creation. If using fork, even numpy arrays will work.
    mp.set_start_method("spawn")

    size = int(np.prod(shape))

    # create a RawArray in shared memory and a numpy array to wrap it
    raw_array = mp.RawArray(ctypes.c_int32, size)
    array = np.frombuffer(raw_array, dtype=np.int32, count=size)
    array.shape = shape

    # if the RawArray is passed as an argument, the child process modifies the
    # array in shared memory
    p = mp.Process(target=fill_raw_array, args=(raw_array,))
    p.start()
    p.join()
    assert np.all(array == 5)

    # reset
    array[:] = 0

    # if the numpy array is passed, a copy is sent to the child process, and the original array is unchanged
    p = mp.Process(target=fill_np_array, args=(array,))
    p.start()
    p.join()
    assert np.all(array == 0)

    # however, Python won't let you send RawArrays through a pipe
    parent_pipe, child_pipe = mp.Pipe()
    p = mp.Process(target=revc_raw_array, args=(child_pipe,))
    p.start()
    try:
        # this will throw an Exception:
        # c_int_Array_25 objects should only be shared between processes through inheritance
        parent_pipe.send(raw_array)
    except Exception as e:
        print(e)
    else:
        assert False, "We know this will throw an exception."
    p.terminate()

def fill_raw_array(raw_array):
    size = int(np.prod(shape))
    array = np.frombuffer(raw_array, dtype=np.int32, count=size)
    array[:] = 5

def fill_np_array(array):
    array[:] = 5

def revc_raw_array(pipe):
    raw_array = pipe.recv()
    raw_array[:] = 5

if __name__ == "__main__":
    main()