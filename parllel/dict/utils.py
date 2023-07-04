from typing import Any, Callable, Iterable, Union

import numpy as np

from parllel.buffers import NamedTuple, NamedTupleClass, Buffer, LeafType

from .array_dict import ArrayDict


def dict_map(
    func: Callable[[ArrayDict, Any], Any],
    arraydict: Union[ArrayDict, tuple],
    *args, **kwargs,
) -> ArrayDict:
    """Call function ``func(buf, *args, **kwargs)`` on all contents of
    ``buffer_``, and return the results.  ``buffer_`` can be an arbitrary
    structure of tuples, namedtuples, namedarraytuples, NamedTuples, and
    NamedArrayTuples, and a new, matching structure will be returned.
    ``None`` fields remain ``None``.
    """
    if isinstance(arraydict, ArrayDict): # non-leaf node
        contents = tuple(dict_map(func, elem, *args, **kwargs) for elem in arraydict)
        if isinstance(arraydict, ArrayDict):
            return arraydict._make(contents)
        # buffer is a tuple
        return contents

    # leaf node
    if arraydict is None:
        return None
    return func(arraydict, *args, **kwargs)


def dict_all(buffer: Buffer, predicate: Callable[[LeafType], bool]) -> bool:
    if isinstance(buffer, tuple): # non-leaf node
        return all(dict_all(elem, predicate) for elem in buffer if elem is not None)

    # leaf node (None elements already filtered out, unless called buffer = None)
    if buffer is None:
        return False
    return predicate(buffer)


def dict_asndarray(arraydict: ArrayDict) -> ArrayDict[np.ndarray]:
    return dict_map(np.asarray, arraydict)


def collate_buffers(buffers: Iterable[NamedTuple], names=None, typename=None):
    """Takes a sequence of NamedTuples and returns an instance of this
    NamedTuple, where the corresponding leaf nodes are from all elements in the
    sequence have been joined into NamedTuples. The new leaf NamedTuples have
    fields according to `names`. If names is not given, buffers must itself be
    NamedTuple of NamedTuples, and names will be the fields of the top-level
    NamedTuple.

    Based on Pytorch default_collate:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    """
    # gets one element even if buffers is NamedArrayTuple
    elem = tuple(buffers)[0]

    # in first call, gets the fields of the top-level NamedTuple
    # recursive calls always just pass on this buffers
    names = names or buffers._fields

    if isinstance(elem, NamedTuple):
        # zip(* pattern results in a transpose
        return elem._make(collate_buffers(matching_elems, names, field)
                            for *matching_elems, field
                            in zip(*buffers, elem._fields))

    # base case
    # the leaves of the new buffer are NamedTuples of the previous leaf nodes
    return NamedTupleClass(typename, names)(*buffers)