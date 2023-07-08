import os
from multiprocessing import shared_memory
from typing import Any
import weakref

import numpy as np

import parllel.logger as logger
from .array import Array


class ManagedMemoryArray(Array, storage="managed"):
    """An array in OS shared memory that can be shared between processes at any
    time.
    """

    storage = "managed"

    def _allocate(self) -> None:
        # allocate array in OS shared memory
        size = int(np.prod(self._base_shape))
        nbytes = size * np.dtype(self.dtype).itemsize

        # keep track of which process created this (presumably the main process)
        self._spawning_process = os.getpid()

        # SharedMemory is given a unique name that other processes can use to
        # attach to it.
        self._shmem = shared_memory.SharedMemory(create=True, size=nbytes)
        logger.debug(
            f"Process {self._spawning_process} allocated {self._shmem.size} "
            f"bytes of shared memory with name {self._shmem.name}"
        )

        # create a finalizer to ensure that shared memory is cleaned up
        self._finalizer = weakref.finalize(self, self._cleanup_shmem)

        self._wrap_raw_array()

    def _cleanup_shmem(self) -> None:
        # close must be called by each instance (i.e. each process) on cleanup
        # calling close on shared memory that has already been unlinked appears
        # not to throw an error
        self._shmem.close()
        # these debug statements may not print if the finalizer is called during
        # process shutdown
        logger.debug(f"Process {os.getpid()} closed shared memory {self._shmem.name}")
        if os.getpid() == self._spawning_process:
            # unlink must be called once and only once to release shared memory
            self._shmem.unlink()
            logger.debug(f"Process {os.getpid()} unlinked shared memory {self._shmem.name}")

    def _wrap_raw_array(self) -> None:
        self._base_array = np.ndarray(
            shape=self._base_shape,
            dtype=self.dtype,
            buffer=self._shmem.buf,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # finalizers are marked dead when transferred between processes
        del state["_finalizer"]
        # remove arrays which cannot be pickled
        del state["_base_array"]
        # subprocesses should not be able to call rotate()
        # if processes are started by fork, this is not guaranteed to be called
        state["_rotatable"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # restore state dict entries
        self.__dict__.update(state)
        # recreate finalizer
        self._finalizer = weakref.finalize(self, self._cleanup_shmem)
        # restore _base_array
        self._wrap_raw_array()

    def close(self):
        self._finalizer()
