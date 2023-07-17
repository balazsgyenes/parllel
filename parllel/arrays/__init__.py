# dependencies
import parllel.buffers

from .array import Array

# these types are only imported so they can be registered
from .jagged import JaggedArray
from .managedmemory import ManagedMemoryArray
from .sharedmemory import SharedMemoryArray


__all__ = [
    "Array",
]
