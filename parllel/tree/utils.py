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


def transpose_dict(d: dict[str, dict[str, Any]]):
    """Accepts a dictionary of dictionaries and returns a transposed dictionary,
    where the top-level keys have been swapped with the nested keys. All nested
    dictionaries are required to have the same keys.
    e.g.
    transpose_dict(
    {
        "agent1": {"value": 1, "dist_params": 2},
        "agent2": {"value": 3, "dist_params": 4},
    }
    )
    ->
    {
        "value": {"agent1": 1, "agent2": 3},
        "dist_params": {"agent1": 2, "agent2": 4},
    }
    """
    subkeys = tuple(d.values())[0].keys()
    return {
        inner_key: {outer_key: subdict[inner_key] for outer_key, subdict in d.items()}
        for inner_key in subkeys
    }


def assert_dict_equal(ref, test, /, name=""):
    if isinstance(ref, Mapping): # non-leaf node
        for key, ref_value in ref.items():
            test_value = test[key]
            assert_dict_equal(ref_value, test_value, name + "." + key)
    else:
        assert np.array_equal(ref, test), name
