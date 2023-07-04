from __future__ import annotations

import dataclasses
from collections.abc import MutableMapping
from operator import getitem
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar, Union

from parllel.dict import ArrayLike

ArrayType = TypeVar("ArrayType", bound=ArrayLike)
NodeType = Union[ArrayType, "ArrayDict[ArrayType]", None]
ValueType = Union[ArrayType, "ArrayDict[ArrayType]", dict[str, "ValueType"], None]


class ArrayDict(MutableMapping, Generic[ArrayType]):
    """A batched dictionary of array-like objects.

    ArrayDict is a container for array-like objects (e.g. np.ndarray,
    torch.Tensor, parllel Arrays, etc.) that are stored as key-value pairs,
    where all arrays have leading batch dimensions in common.

    This class is heavily inspired by torch's TensorDict.
    """

    def __init__(
        self,
        items: dict[str, ValueType] | Iterable[tuple[str, ValueType]],
        /,
    ) -> None:
        # TODO: clean items to ensure only leaf nodes or ArrayDicts

        self._dict: dict[str, NodeType] = dict(items)

    def __getitem__(self, key: Any) -> NodeType:
        if isinstance(key, str):
            return self._dict[key]

        try:
            return ArrayDict(
                (field, arr[key] if arr is not None else None)
                for field, arr in self._dict.items()
            )
        except IndexError as e:
            for field, arr in self._dict.items():
                try:
                    _ = arr[key] if arr is not None else None
                except IndexError:
                    raise IndexError(
                        f"Index error in field '{field}' for index '{key}'"
                    ) from e

    def __setitem__(self, key: Any, value: ValueType) -> None:
        if isinstance(key, str):
            self._dict[key] = value
            return

        if isinstance(value, MutableMapping):  # i.e. dict, ArrayDict, etc.
            getter = getitem
        elif dataclasses.is_dataclass(value):
            getter = getattr
        else:
            # don't index into scalars, just assign the same scalar to all
            # fields
            getter = lambda obj, field: obj

        for field, arr in self._dict.items():
            subvalue = getter(value, field)

            if arr is not None and subvalue is not None:
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

    def __getattr__(self, name: str) -> ArrayAttrDict:
        try:
            return ArrayAttrDict(
                (
                    (field, getattr(arr, name) if arr is not None else None)
                    for field, arr in self._dict.items()
                ),
                name=name,
            )
        except AttributeError as e:
            for field, arr in self._dict.items():
                try:
                    _ = getattr(arr, name) if arr is not None else None
                except IndexError:
                    raise IndexError(
                        f"Attribute error in field '{field}' for attribute '{name}'"
                    ) from e
            raise e

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def batch_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def apply(self, fn: Callable, **kwargs) -> ArrayDict:
        items = []
        for field, arr in self._dict.items():
            if isinstance(arr, ArrayDict):  # non-leaf node
                items.append((field, arr.apply(fn, **kwargs)))

            # leaf node
            else:
                items.append((field, fn(arr, **kwargs) if arr is not None else None))

        return ArrayDict(items)


class ArrayAttrDict(ArrayDict):
    def __init__(self, *args, name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        items = []
        for field, method in self._dict.items():
            try:
                result = method(*args, **kwds) if method is not None else None
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