from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
from multiprocessing.shared_memory import SharedMemory as MpSharedMem
from typing import Any, Literal
from weakref import WeakMethod, finalize

import numpy as np

import parllel.logger as logger

StorageType = Literal["local", "shared", "inherited"]


class Storage:
    """A block of memory that exposes a memoryview, which can be wrapped with
    an ndarray.
    """

    _subclasses: dict[tuple[str, bool], type[Storage]] = {}
    kind: StorageType
    resizable: bool
    _shape: tuple[int, ...]
    _dtype = np.dtype

    def __init_subclass__(
        cls,
        *,
        kind: str,
        resizable: bool | None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if resizable is not None:
            cls._subclasses[(kind, resizable)] = cls
        else:
            cls._subclasses[(kind, False)] = cls
            cls._subclasses[(kind, True)] = cls

    def __new__(
        cls,
        *args,
        kind: str | None = None,
        resizable: bool = False,
        **kwargs,
    ) -> Storage:
        if kind is None:
            if cls is Storage:
                # forbidden to instantiate parent class directly
                raise TypeError("Storage requires kind to be specified.")
            subcls = cls
        else:
            try:
                subcls = cls._subclasses[(kind, resizable)]
            except KeyError:
                raise ValueError(
                    f"No array subclass registered under {kind=} and {resizable=}"
                )
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
    def size(self) -> int:
        return int(np.prod(self._shape))

    def __enter__(self) -> np.ndarray:
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def get_numpy(self) -> np.ndarray:
        raise NotImplementedError

    def resize(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        raise TypeError(f"Not possible to resize {type(self).__name__}.")

    def close(self, force: bool = False) -> None:
        ...


class LocalMemory(Storage, kind="local", resizable=None):
    kind = "local"
    resizable = True

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


def call_weakmethod(weakmethod: WeakMethod, *args, **kwargs) -> None:
    method = weakmethod()
    if method is not None:
        return method(*args, **kwargs)


class SharedMemory(Storage, kind="shared", resizable=False):
    """An array in OS shared memory that can be shared between processes at any
    time.
    """

    kind = "shared"
    resizable = False
    _shmem: MpSharedMem

    def __init__(
        self,
        kind: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        resizable: bool = False,
    ) -> None:
        self._shape = shape
        self._dtype = dtype

        nbytes = self.size * np.dtype(dtype).itemsize
        self._shmem = MpSharedMem(create=True, size=nbytes)

        logger.debug(
            f"Process {os.getpid()} allocated {self._shmem.size} bytes of shared "
            f"memory with name {self._shmem.name}"
        )

        # keep track of which process created this (presumably the main process)
        self._spawning_pid = os.getpid()

        # create a finalizer to ensure that shared memory is cleaned up
        self._finalizer = finalize(self, call_weakmethod, WeakMethod(self.close))

    def __enter__(self) -> np.ndarray:
        return np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self._shmem.buf,
        )

    def get_numpy(self) -> np.ndarray:
        return np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self._shmem.buf,
        )

    def __setstate__(self, state: dict) -> None:
        # restore state dict entries
        self.__dict__.update(state)

        # finalizer is marked dead when transferred between processes
        # restore finalizer by recreating it
        self._finalizer = finalize(self, call_weakmethod, WeakMethod(self.close))

    def close(self, force: bool = False) -> None:
        # close must be called by each instance (i.e. each process) on cleanup
        # calling close on shared memory that has already been unlinked appears
        # not to throw an error
        self._shmem.close()
        pid = os.getpid()
        if force or pid == self._spawning_pid:
            # unlink must be called once and only once to release shared memory
            self._shmem.unlink()
            # this debug statement may not print if the finalizer is called during
            # process shutdown
            logger.debug(f"Process {pid} unlinked shared memory {self._shmem.name}")


class ResizableSharedMemory(Storage, kind="shared", resizable=True):
    """An array in OS shared memory that can be shared between processes at any
    time.
    """

    kind = "shared"
    resizable = True
    _name: str

    def __init__(
        self,
        kind: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        resizable: bool = True,
    ) -> None:
        self._shape = shape
        self._dtype = dtype

        nbytes = self.size * np.dtype(dtype).itemsize
        shmem = MpSharedMem(create=True, size=nbytes)
        # SharedMemory is given a unique name that other processes can use to
        # attach to it.
        self._name = shmem.name
        # create a dictionary to store hard references to shmem until the
        # ndarrays using them are garbage collected
        self._hard_refs: dict[int, tuple[MpSharedMem, finalize]] = {}

        logger.debug(
            f"Process {os.getpid()} allocated {shmem.size} bytes of shared "
            f"memory with name {shmem.name}"
        )

        # keep track of which process created this (presumably the main process)
        self._spawning_pid = os.getpid()

    def __enter__(self) -> np.ndarray:
        # store a reference to the SharedMem so that it doesn't get cleaned up
        self._shmem = MpSharedMem(create=False, name=self._name)
        return np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self._shmem.buf,
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._shmem.close()
        # delete hard reference to SharedMem allowing it to be garbage collected
        del self._shmem
        return False

    def get_numpy(self) -> np.ndarray:
        shmem = MpSharedMem(create=False, name=self._name)
        ndarray = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=shmem.buf)
        finalizer = finalize(ndarray, self._cleanup_numpy, id(ndarray))
        # lookup ndarray by id, which also avoids storing a reference to ndarray
        # and preventing it from being garbage collected
        self._hard_refs[id(ndarray)] = (shmem, finalizer)
        return ndarray

    def _cleanup_numpy(self, id_ndarray: int) -> None:
        shmem, _ = self._hard_refs.pop(id_ndarray)
        shmem.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_hard_refs"] = {}
        return state

    def close(self, force: bool = False) -> None:
        # close must be called by each instance (i.e. each process) on cleanup
        # calling close on shared memory that has already been unlinked appears
        # not to throw an error
        for shmem, _ in self._hard_refs.values():
            shmem.close()
        pid = os.getpid()
        if force or pid == self._spawning_pid:
            # unlink must be called once and only once to release shared memory
            shmem = MpSharedMem(create=False, name=self._name)
            # TODO: this isn't the right name, because the prefix is gone
            shmem.close()
            shmem.unlink()
            logger.debug(f"Process {pid} unlinked shared memory {shmem.name}")


class InheritedMemory(Storage, kind="inherited", resizable=False):
    """An array in OS shared memory that can be shared between processes on
    process startup only (i.e. process inheritance). Starting processes with
    the `spawn` method is also supported.
    """

    kind = "inherited"
    resizable = False

    def __init__(
        self,
        kind: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        resizable: bool = False,
    ) -> None:
        self._shape = shape
        self._dtype = dtype
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
