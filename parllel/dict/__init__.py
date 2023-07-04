from typing import Any, Protocol, Union


class ArrayLike(Protocol):
    def __getitem__(self, key: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...


from .array_dict import ArrayDict

ArrayTree = Union[ArrayLike, ArrayDict, None]


__all__ = [
    "ArrayLike",
    "ArrayDict",
    "ArrayTree",
]
