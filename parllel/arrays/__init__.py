# dependencies
import parllel.tree

from .array import Array
from .indices import Index, Location

# isort: split
# these modules are only imported to register the types they define
import parllel.arrays.jagged
import parllel.arrays.managedmemory
import parllel.arrays.sharedmemory

__all__ = [
    "Array",
    "Index",
    "Location",
]
