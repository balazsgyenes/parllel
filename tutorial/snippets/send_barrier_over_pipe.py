"""This snippet demonstrates that multiprocessing.Barrier cannot be passed to
another process after startup.
"""

import multiprocessing as mp
from multiprocessing.connection import Connection


def worker(pipe: Connection):
    barrier: mp.Barrier = pipe.recv()
    print("Received barrier")
    barrier.wait()
    print("Worker passed barrier")


if __name__ == "__main__":
    # when creating processes using spawn, resources must be pickled to the
    # child process after creation. If using fork, even numpy arrays will work.
    mp.set_start_method("spawn")

    parent_pipe, child_pipe = mp.Pipe()

    # if the RawArray is passed as an argument, the child process modifies the
    # array in shared memory
    p = mp.Process(target=worker, args=(child_pipe,))
    p.start()

    barrier = mp.Barrier(parties=2)
    parent_pipe.send(barrier)
    print("Sent barrier")
    barrier.wait()

    p.join()
