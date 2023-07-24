from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Iterable

import numpy as np

from parllel.tree import ArrayDict, ArrayLike, ArrayTree, ArrayType, ArrayOrMapping


def dict_map(
    func: Callable[[ArrayType], Any],
    tree: ArrayOrMapping[ArrayType],
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


def assert_dict_equal(ref, test, /, name=""):
    if isinstance(ref, Mapping): # non-leaf node
        for key, ref_value in ref.items():
            test_value = test[key]
            assert_dict_equal(ref_value, test_value, name + "." + key)
    else:
        assert np.array_equal(ref, test), name
