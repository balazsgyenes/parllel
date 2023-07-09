import os
from multiprocessing.shared_memory import SharedMemory
from typing import Any
from weakref import finalize

import numpy as np

import parllel.logger as logger
from .array import Array


class ManagedMemoryArray(Array, storage="managed"):
    """An array in OS shared memory that can be shared between processes at any
    time.
    """

    storage = "managed"

    def _allocate(self, shape: tuple[int, ...], dtype: np.dtype, name: str) -> None:
        # allocate array in OS shared memory
        size = int(np.prod(shape))
        nbytes = size * np.dtype(dtype).itemsize

        # keep track of which process created this (presumably the main process)
        spawning_pid = os.getpid()

        # SharedMemory is given a unique name that other processes can use to
        # attach to it.
        shmem = SharedMemory(create=True, size=nbytes)
        logger.debug(
            f"Process {spawning_pid} allocated {shmem.size} bytes of shared "
            f"memory with name {shmem.name}"
        )

        # create a finalizer to ensure that shared memory is cleaned up
        finalizer = finalize(self, self._cleanup_shmem, shmem, spawning_pid)

        # store allocation for getstate and setstate methods
        if not hasattr(self, "_allocations"):
            self._allocations = []
            self._finalizers = []
        self._allocations.append((name, shmem, shape, dtype, spawning_pid))
        self._finalizers.append(finalizer)

        array = self._wrap_with_ndarray(shmem, shape, dtype)

        # assign to requested attribute name
        setattr(self, name, array)

    @staticmethod
    def _cleanup_shmem(shared_mem: SharedMemory, spawning_pid: int) -> None:
        # close must be called by each instance (i.e. each process) on cleanup
        # calling close on shared memory that has already been unlinked appears
        # not to throw an error
        shared_mem.close()
        # these debug statements may not print if the finalizer is called during
        # process shutdown
        logger.debug(f"Process {os.getpid()} closed shared memory {shared_mem.name}")
        if os.getpid() == spawning_pid:
            # unlink must be called once and only once to release shared memory
            shared_mem.unlink()
            logger.debug(f"Process {os.getpid()} unlinked shared memory {shared_mem.name}")

    @staticmethod
    def _wrap_with_ndarray(
        shared_mem: SharedMemory,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray:
        return np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # subprocesses should not be able to call rotate()
        # if processes are started by fork, this is not guaranteed to be called
        state["_rotatable"] = False

        # finalizers are marked dead when transferred between processes
        del state["_finalizers"]

        # remove allocated numpy arrays which cannot be pickled
        for allocation in self._allocations:
            name = allocation[0]
            del state[name]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # restore state dict entries
        self.__dict__.update(state)

        # restore numpy arrays by re-wrapping them
        # restore finalizers by recreating them
        self._finalizers = []
        for allocation in self._allocations:
            name, shmem, shape, dtype, spawning_pid = allocation
            array = self._wrap_with_ndarray(shmem, shape, dtype)
            setattr(self, name, array)
            finalizer = finalize(self, self._cleanup_shmem, shmem, spawning_pid)
            self._finalizers.append(finalizer)
        
    def close(self):
        for finalizer in self._finalizers:
            finalizer()
