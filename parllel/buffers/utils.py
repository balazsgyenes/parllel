from typing import Any, Callable, Union

from parllel.buffers import Buffer
from .named_tuple import NamedTuple


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


def buffer_func(func: Callable[[Buffer, Any], Any], buffer: Union[Buffer, tuple],
                *args, **kwargs) -> Buffer:
    """Call function ``func(buf, *args, **kwargs)`` on all contents of
    ``buffer_``, and return the results.  ``buffer_`` can be an arbitrary
    structure of tuples, namedtuples, namedarraytuples, NamedTuples, and
    NamedArrayTuples, and a new, matching structure will be returned.
    ``None`` fields remain ``None``.
    """
    if isinstance(buffer, tuple): # non-leaf node
        contents = tuple(buffer_func(func, elem, *args, **kwargs) for elem in buffer)
        if isinstance(buffer, NamedTuple):
            return buffer._make(contents)
        # buffer is a tuple
        return contents

    # leaf node
    if buffer is None:
        return None
    return func(buffer, *args, **kwargs)
