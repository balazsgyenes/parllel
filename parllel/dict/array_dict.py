from __future__ import annotations

from collections import UserDict
from typing import Any


class ArrayDict(UserDict):
    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return super().__getitem__(key)
        
        try:
            return ArrayDict(
                (field, arr[key] if arr is not None else None)
                for field, arr in self.items()
            )
        except IndexError as e:
            for field, arr in self.items():
                try:
                    _ = arr[key] if arr is not None else None
                except IndexError:
                    raise IndexError(
                        f"Index error in field '{field}' for index '{key}'"
                    ) from e

    def __getattr__(self, name: str) -> ArrayDict:
        try:
            return ArrayAttrDict(name,(
                (field, getattr(arr, name) if arr is not None else None)
                for field, arr in self.items()
            ))
        except AttributeError as e:
            for field, arr in self.items():
                try:
                    _ = getattr(arr, name) if arr is not None else None
                except IndexError:
                    raise IndexError(
                        f"Attribute error in field '{field}' for attribute '{name}'"
                    ) from e

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(key, str):
            return super().__setitem__(key, value)

        is_dict = isinstance(value, (ArrayDict, dict))

        for field, arr in self.items():
            
            # don't index into scalars, just assign the same scalar to all
            # fields
            subvalue = value[field] if is_dict else value

            if arr is not None and subvalue is not None:
                try:
                    arr[key] = subvalue
                except IndexError as e:
                    raise IndexError(
                        f"Index error in field '{field}' for index '{key}'"
                    ) from e

    # def __setattr__(self, name: str, value: Any) -> None:
    #     is_dict = isinstance(value, (ArrayDict, dict))
        
    #     for field, arr in self.items():
    #         # don't index into scalars, just assign the same scalar to all
    #         # fields
    #         subvalue = value[field] if is_dict else value

    #         if arr is not None and subvalue is not None:
    #             try:
    #                 setattr(arr, name, subvalue)
    #             except AttributeError as e:
    #                 raise AttributeError(
    #                     f"Attribute error in field {field} for attribute {name}"
    #                 ) from e


class ArrayAttrDict(ArrayDict):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        new_items = []

        for field, method in self.items():

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

            new_items.append((field, result))
        
        return ArrayDict(new_items)
