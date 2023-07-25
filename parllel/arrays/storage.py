from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
from multiprocessing.shared_memory import SharedMemory as MpSharedMem
from typing import Literal
from weakref import finalize

import numpy as np

import parllel.logger as logger

StorageType = Literal["local", "shared", "inherited"]


class Storage:
    """A block of memory that exposes a memoryview, which can be wrapped with
    an ndarray.
    """

    _subclasses = {}
    kind: StorageType

    def __init_subclass__(
        cls,
        *,
        kind: str | None = None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)
        kind = kind if kind is not None else "local"
        cls._subclasses[kind] = cls

    def __new__(cls, *args, kind: str, **kwargs) -> Storage:
        # note that Storage object can never be instantiated, only its
        # subclasses
        try:
            subcls = cls._subclasses[kind]
        except KeyError:
            raise ValueError(f"No array subclass registered under {kind=}")
        return super().__new__(subcls)

    def __init__(self, kind: str, size: int, dtype: np.dtype) -> None:
        self._size = size
        self._allocate(size=size, dtype=dtype)

    @property
    def size(self) -> int:
        return self._size

    @property
    def buffer(self) -> memoryview:
        raise NotImplementedError

    def _allocate(self, size: int, dtype: np.dtype) -> None:
        raise NotImplementedError

    def resize(self, size: int, dtype: np.dtype) -> None:
        self.close()
        self._allocate(size=size, dtype=dtype)

    def close(self) -> None:
        ...


class LocalMemory(Storage, kind="local"):
    kind = "local"

    def _allocate(self, size: int, dtype: np.dtype) -> None:
        self._memory = np.zeros((size,), dtype=dtype)

    @property
    def buffer(self) -> memoryview:
        return self._memory.data


class SharedMemory(Storage, kind="shared"):
    """An array in OS shared memory that can be shared between processes at any
    time.
    """

    kind = "shared"

    def __init__(self, kind: str, size: int, dtype: np.dtype) -> None:
        super().__init__(kind=kind, size=size, dtype=dtype)

        # keep track of which process created this (presumably the main process)
        self._spawning_pid = os.getpid()

        # create a finalizer to ensure that shared memory is cleaned up
        self._finalizer = finalize(self, self.close)

    def _allocate(self, size: int, dtype: np.dtype, name: str | None = None) -> None:
        nbytes = size * np.dtype(dtype).itemsize
        shmem = MpSharedMem(create=True, name=name, size=nbytes)
        # SharedMemory is given a unique name that other processes can use to
        # attach to it.
        self._name = shmem.name
        logger.debug(
            f"Process {os.getpid()} allocated {shmem.size} bytes of shared "
            f"memory with name {shmem.name}"
        )

    def _get_shmem(self) -> MpSharedMem:
        # this operation takes around 10us
        # by doing this only on demand, we ensure that resizing operations
        # take effect
        return MpSharedMem(create=False, name=self._name)

    @property
    def buffer(self) -> memoryview:
        return self._get_shmem().buf

    def resize(self, size: int, dtype: np.dtype) -> None:
        self.close(force=True)
        self._allocate(size=size, dtype=dtype, name=self._name)

    def __setstate__(self, state: dict) -> None:
        # restore state dict entries
        self.__dict__.update(state)

        # finalizer is marked dead when transferred between processes
        # restore finalizer by recreating it
        self._finalizer = finalize(self, self.close)

    def close(self, force: bool = False) -> None:
        # close must be called by each instance (i.e. each process) on cleanup
        # calling close on shared memory that has already been unlinked appears
        # not to throw an error
        shmem = self._get_shmem()
        pid = os.getpid()
        shmem.close()
        # these debug statements may not print if the finalizer is called during
        # process shutdown
        logger.debug(f"Process {pid} closed shared memory {self._name}")
        if force or pid == self._spawning_pid:
            # unlink must be called once and only once to release shared memory
            shmem.unlink()
            logger.debug(f"Process {pid} unlinked shared memory {self._name}")


class InheritedMemory(Storage, kind="inherited"):
    """An array in OS shared memory that can be shared between processes on
    process startup only (i.e. process inheritance). Starting processes with
    the `spawn` method is also supported.
    """

    kind = "inherited"

    def _allocate(self, size: int, dtype: np.dtype) -> None:
        # allocate array in OS shared memory
        nbytes = size * np.dtype(dtype).itemsize
        # mp.RawArray can be safely passed between processes on startup, even
        # when using the "spawn" start method. However, it cannot be sent
        # through a Pipe or Queue
        self._raw_array = mp.RawArray(ctypes.c_char, nbytes)

    def buffer(self) -> memoryview:
        # TODO: there must be a better way to do this
        # if you call the ndarray constructor with raw_array.raw, the resulting
        # array is readonly
        # this might also work, but seems hacky:
        # return self._raw_array._wrapper.create_memoryview()
        array: np.ndarray = np.frombuffer(
            self._raw_array,
            dtype=np.uint8,
            count=len(self._raw_array),
        )
        return array.data
