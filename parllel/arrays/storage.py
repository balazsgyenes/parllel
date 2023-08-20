from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
from multiprocessing.shared_memory import SharedMemory as MpSharedMem
from typing import Any, Literal
from weakref import finalize

import numpy as np

import parllel.logger as logger

StorageType = Literal["local", "shared", "inherited"]


class Storage:
    """A block of memory that exposes a memoryview, which can be wrapped with
    an ndarray.
    """

    _subclasses = {}
    _shape: tuple[int, ...]
    _dtype = np.dtype
    _resizable: bool
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

    def __new__(cls, *args, kind: str | None = None, **kwargs) -> Storage:
        if kind is None:
            if cls is Storage:
                # forbidden to instantiate parent class directly
                raise TypeError("Storage requires kind to be specified.")
            subcls = cls
        else:
            try:
                subcls = cls._subclasses[kind]
            except KeyError:
                raise ValueError(f"No array subclass registered under {kind=}")
        return super().__new__(subcls)

    def __init__(
        self,
        kind: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        resizable: bool = False,
    ) -> None:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def resizable(self) -> bool:
        return self._resizable

    @property
    def size(self) -> int:
        return int(np.prod(self._shape))

    def __enter__(self) -> np.ndarray:
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def get_numpy(self) -> np.ndarray:
        raise NotImplementedError

    def resize(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        raise NotImplementedError

    def close(self) -> None:
        ...


class LocalMemory(Storage, kind="local"):
    kind = "local"

    def __init__(
        self,
        kind: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        resizable: bool = True,
    ) -> None:
        self._shape = shape
        self._dtype = dtype
        self._resizable = True
        self._memory = np.zeros(shape, dtype)

    def resize(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        new_memory = np.zeros(shape, dtype)

        # copy data to new memory
        location = tuple(slice(0, size) for size in self.shape)
        new_memory[location] = self._memory

        self._memory = new_memory
        self._shape = shape
        self._dtype = dtype

    def __enter__(self):
        return self._memory

    def get_numpy(self) -> np.ndarray:
        return self._memory


class SharedMemory(Storage, kind="shared"):
    """An array in OS shared memory that can be shared between processes at any
    time.
    """

    kind = "shared"
    _shmem: MpSharedMem
    _name: str

    def __init__(
        self,
        kind: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        resizable: bool = False,
    ) -> None:
        self._shape = shape
        self._dtype = dtype
        self._resizable = resizable

        nbytes = self.size * np.dtype(dtype).itemsize
        shmem = MpSharedMem(create=True, size=nbytes)
        if self.resizable:
            # SharedMemory is given a unique name that other processes can use to
            # attach to it.
            self._name = shmem.name
            # create a dictionary to store hard references to shmem until the
            # ndarrays using them are garbage collected
            self._hard_refs: dict[int, tuple[MpSharedMem, finalize]] = {}
        else:
            self._shmem = shmem

        logger.debug(
            f"Process {os.getpid()} allocated {shmem.size} bytes of shared "
            f"memory with name {shmem.name}"
        )

        # keep track of which process created this (presumably the main process)
        self._spawning_pid = os.getpid()

        # create a finalizer to ensure that shared memory is cleaned up
        self._finalizer = finalize(self, self.close)

    def __enter__(self) -> np.ndarray:
        if self.resizable:
            # store a reference to the SharedMem so that it doesn't get cleaned up
            self._shmem = MpSharedMem(create=False, name=self._name)

        return np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self._shmem.buf,
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.resizable:
            self._shmem.close()
            # delete hard reference to SharedMem allowing it to be garbage collected
            del self._shmem
        return False

    def get_numpy(self) -> np.ndarray:
        if self.resizable:
            shmem = MpSharedMem(create=False, name=self._name)
            ndarray = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=shmem.buf)
            # finalizer = finalize(ndarray, self._hard_refs.pop, id(ndarray))
            finalizer = finalize(ndarray, self._hard_refs.pop, id(ndarray))
            # lookup ndarray by id, which also avoids storing a reference to ndarray
            # and preventing it from being garbage collected
            self._hard_refs[id(ndarray)] = (shmem, finalizer)
            return ndarray
        else:
            return np.ndarray(
                shape=self.shape,
                dtype=self.dtype,
                buffer=self._shmem.buf,
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_hard_refs", None)
        return state

    def __setstate__(self, state: dict) -> None:
        # restore state dict entries
        self.__dict__.update(state)

        if self.resizable:
            self._hard_refs = {}

        # finalizer is marked dead when transferred between processes
        # restore finalizer by recreating it
        self._finalizer = finalize(self, self.close)

    def close(self, force: bool = False) -> None:
        # close must be called by each instance (i.e. each process) on cleanup
        # calling close on shared memory that has already been unlinked appears
        # not to throw an error
        if self.resizable:
            shmem = MpSharedMem(create=False, name=self._name)
        else:
            shmem = self._shmem
        shmem.close()
        pid = os.getpid()
        if force or pid == self._spawning_pid:
            # unlink must be called once and only once to release shared memory
            shmem.unlink()
            # this debug statement may not print if the finalizer is called during
            # process shutdown
            logger.debug(f"Process {pid} unlinked shared memory {shmem.name}")


class InheritedMemory(Storage, kind="inherited"):
    """An array in OS shared memory that can be shared between processes on
    process startup only (i.e. process inheritance). Starting processes with
    the `spawn` method is also supported.
    """

    kind = "inherited"

    def __init__(
        self,
        kind: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        resizable: bool = False,
    ) -> None:
        self._shape = shape
        self._dtype = dtype
        self._resizable = resizable
        # allocate array in OS shared memory
        nbytes = self.size * np.dtype(dtype).itemsize
        # mp.RawArray can be safely passed between processes on startup, even
        # when using the "spawn" start method. However, it cannot be sent
        # through a Pipe or Queue
        self._raw_array = mp.RawArray(ctypes.c_char, nbytes)
        self._wrap_raw_array()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_ndarray"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # restore state dict entries
        self.__dict__.update(state)
        self._wrap_raw_array()

    def _wrap_raw_array(self) -> None:
        self._ndarray = np.frombuffer(
            self._raw_array,
            dtype=self.dtype,
            count=self.size,
        )
        # assign to shape attribute so that error is raised when data is copied
        # array.reshape might silently copy the data
        self._ndarray.shape = self.shape

    def __enter__(self) -> np.ndarray:
        return self._ndarray

    def get_numpy(self) -> np.ndarray:
        return self._ndarray
