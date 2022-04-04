from typing import Any, Callable, Union

from .buffer import Buffer, LeafType
from .named_tuple import NamedTuple, NamedArrayTuple


def buffer_method(buffer: Union[Buffer, tuple], method_name: str, *args, **kwargs) -> Buffer:
    """Call method ``method_name(*args, **kwargs)`` on all contents of
    ``buffer``, and return the results. ``buffer`` can be an arbitrary
    structure of tuples, namedtuples, namedarraytuples, NamedTuples, and
    NamedArrayTuples, and a new, matching structure will be returned.
    ``None`` fields remain ``None``.
    """
    if isinstance(buffer, tuple): # non-leaf node
        contents = tuple(buffer_method(elem, method_name, *args, **kwargs) for elem in buffer)
        if isinstance(buffer, NamedTuple):
            return buffer._make(contents)
        # buffer is a tuple
        return contents

    # leaf node
    if buffer is None:
        return None
    return getattr(buffer, method_name)(*args, **kwargs)


def buffer_map(func: Callable[[Buffer, Any], Any], buffer: Union[Buffer, tuple],
                *args, **kwargs) -> Buffer:
    """Call function ``func(buf, *args, **kwargs)`` on all contents of
    ``buffer_``, and return the results.  ``buffer_`` can be an arbitrary
    structure of tuples, namedtuples, namedarraytuples, NamedTuples, and
    NamedArrayTuples, and a new, matching structure will be returned.
    ``None`` fields remain ``None``.
    """
    if isinstance(buffer, tuple): # non-leaf node
        contents = tuple(buffer_map(func, elem, *args, **kwargs) for elem in buffer)
        if isinstance(buffer, NamedTuple):
            return buffer._make(contents)
        # buffer is a tuple
        return contents

    # leaf node
    if buffer is None:
        return None
    return func(buffer, *args, **kwargs)


def buffer_all(buffer: Buffer, predicate: Callable[[LeafType], bool]) -> bool:
    if isinstance(buffer, tuple): # non-leaf node
        return all(buffer_all(elem, predicate) for elem in buffer if elem is not None)

    # leaf node (None elements already filtered out, unless called buffer = None)
    if buffer is None:
        return False
    return predicate(buffer)


def buffer_replace(buffer: Buffer, other_buffer: Buffer) -> Buffer:
    """Replaces leaf nodes in buffer with leaf nodes in other_buffer, if they
    are present. Leaf nodes not present in other_buffer are left unchanged.
    Buffer must be a superset of other_buffer.
    """
    if isinstance(buffer, NamedTuple): # non-leaf node
        assert isinstance(other_buffer, NamedArrayTuple)
        # recursively replace subelements of buffer in the corresponding field
        other_dict = {k: buffer_replace(getattr(buffer, k), v)
                      for (k, v) in other_buffer._asdict().items()}
        return buffer._replace(**other_dict)

    # leaf node
    return other_buffer


def buffer_rotate(buffer: Union[Buffer, tuple]) -> Buffer:
    if isinstance(buffer, tuple): # non-leaf node
        contents = tuple(buffer_rotate(elem) for elem in buffer)
        if isinstance(buffer, NamedTuple): 
            return buffer._make(contents)
        # buffer is a tuple
        return contents

    # leaf node
    if hasattr(buffer, "rotate"):
        buffer.rotate()
    return buffer
