from typing import Any, Protocol, Union


class ArrayLike(Protocol):
    def __getitem__(self, indices: Any) -> Any:
        ...

    def __setitem__(self, indices: Any, value: Any) -> None:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def dtype(self) -> Any:
        ...


from .array_dict import ArrayDict

ArrayTree = Union[ArrayLike, ArrayDict, None]
# TODO: replace with Generic that allows specifying array type

from .utils import dict_map

__all__ = [
    "ArrayLike",
    "ArrayDict",
    "ArrayTree",
    "dict_map",
]
