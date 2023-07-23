from __future__ import annotations

import dataclasses
from collections.abc import Mapping, MutableMapping
from operator import getitem
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar

import numpy as np

from parllel.tree import ArrayLike, ArrayTree, ArrayType, MappingTree

_T = TypeVar("_T")


class ArrayDict(MutableMapping, Generic[ArrayType]):
    """A batched dictionary of array-like objects.

    ArrayDict is a container for array-like objects (e.g. np.ndarray,
    torch.Tensor, parllel Arrays, etc.) that are stored as key-value pairs,
    where all arrays have leading batch dimensions in common.

    This class is heavily inspired by torch's TensorDict.
    """

    def __init__(
        self,
        items: MappingTree | Iterable[tuple[str, MappingTree]] | None = None,
        _run_checks: bool = True,
    ) -> None:
        dict_ = dict(items) if items is not None else {}

        if _run_checks:
            # clean tree to ensure only leaf nodes or ArrayDicts
            # this also shallow copies all the Mapping objects without copying
            # the leaf nodes
            for key, value in dict_.items():
                if isinstance(value, Mapping):
                    dict_[key] = ArrayDict(value)

        self._dict: dict[str, ArrayTree[ArrayType]] = dict_

    def get(self, key: str, default: _T = None) -> ArrayTree[ArrayType] | _T:
        return self._dict.get(key, default)

    def __getitem__(self, key: Any) -> ArrayTree[ArrayType]:
        if isinstance(key, str):
            return self._dict[key]

        try:
            return ArrayDict(
                ((field, arr[key]) for field, arr in self._dict.items()),
                _run_checks=False,
            )
        except IndexError as e:
            for field, arr in self._dict.items():
                try:
                    _ = arr[key]
                except IndexError:
                    raise IndexError(
                        f"Index error in field '{field}' for index '{key}'"
                    ) from e
            raise e

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            self._dict[key] = value
            return

        if isinstance(value, Mapping):  # i.e. dict, ArrayDict, etc.
            getter = getitem
        elif dataclasses.is_dataclass(value):
            getter = getattr
        else:
            # don't index into scalars, just assign the same scalar to all
            # fields
            getter = lambda obj, field: obj

        for field, arr in self._dict.items():
            subvalue = getter(value, field)
            try:
                arr[key] = subvalue
            except IndexError as e:
                raise IndexError(
                    f"Index error in field '{field}' for index '{key}'"
                ) from e

    def __delitem__(self, __key: str) -> None:
        if isinstance(__key, str):
            del self._dict[__key]
        else:
            raise IndexError(f"Cannot delete index {__key}")

    def __iter__(self) -> Iterator:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return repr(self._dict)

    def __getattr__(self, name: str) -> ArrayAttrDict:
        try:
            return ArrayAttrDict(
                ((field, getattr(arr, name)) for field, arr in self._dict.items()),
                name=name,
                _run_checks=False,
                # _run_checks=True would converted nested ArrayAttrDicts back
                # to ArrayDict
            )
        except AttributeError as e:
            for field, arr in self._dict.items():
                try:
                    _ = getattr(arr, name)
                except IndexError:
                    raise IndexError(
                        f"Attribute error in field '{field}' for attribute '{name}'"
                    ) from e
            raise e

    def __getstate__(self) -> dict[str, Any]:
        # define getstate and setstate explicitly so that pickle does not
        # use getattr method, which results in a recursive loop
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def batch_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def apply(self, fn: Callable, **kwargs) -> ArrayDict:
        return ArrayDict(
            (
                (
                    field,
                    (
                        arr.apply(fn, **kwargs)
                        if isinstance(arr, ArrayDict)
                        else fn(arr, **kwargs)
                    ),
                )
                for field, arr in self.items()
            ),
            _run_checks=False,
        )

    def to_ndarray(self) -> ArrayDict[np.ndarray]:
        return self.apply(to_ndarray)

    map = apply


def to_ndarray(leaf: ArrayLike) -> np.ndarray | ArrayDict[np.ndarray]:
    if hasattr(leaf, "to_ndarray"):
        return leaf.to_ndarray()
    return np.asarray(leaf)


class ArrayAttrDict(ArrayDict):
    def __init__(self, *args, name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        items = []
        for field, method in self._dict.items():
            try:
                result = method(*args, **kwds)
            except Exception as e:
                if not callable(method):
                    raise RuntimeError(
                        f"Attribute '{self.name}' of field '{field}' is not callable!"
                    ) from e

                raise RuntimeError(
                    f"Exception from calling method '{self.name}' of field '{field}'"
                ) from e

            items.append((field, result))

        return ArrayDict(items)
