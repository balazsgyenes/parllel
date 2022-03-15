from .array import Array
from .rotating import RotatingArray
from .sharedmemory import SharedMemoryArray, RotatingSharedMemoryArray
from .managedmemory import ManagedMemoryArray, RotatingManagedMemoryArray

__all__ = [
    Array,
    RotatingArray,
    SharedMemoryArray, RotatingSharedMemoryArray,
    ManagedMemoryArray, RotatingManagedMemoryArray,
]

"""
TODO: maybe merge SharedMemoryArray into Array, potentially with the ability to move an array into shared memory if it isn't there already
"""