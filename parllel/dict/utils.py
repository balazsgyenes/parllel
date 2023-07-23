from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Iterable

from parllel.dict import ArrayDict, ArrayLike, ArrayTree, ArrayType, MappingTree


def dict_map(
    func: Callable[[ArrayType], Any],
    tree: MappingTree[ArrayType],
    *args,
    **kwargs,
) -> ArrayTree:
    if isinstance(tree, Mapping):  # non-leaf node
        return ArrayDict(
            ((field, dict_map(func, arr, *args, **kwargs)) for field, arr in tree.items())
        )
    # leaf node
    return func(tree, *args, **kwargs)


def dict_all(tree: ArrayTree, predicate: Callable[[ArrayLike], bool]) -> bool:
    if isinstance(tree, Mapping):  # non-leaf node
        return all(dict_all(elem, predicate) for elem in tree)

    # leaf node
    return predicate(tree)


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
