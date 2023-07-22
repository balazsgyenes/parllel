from typing import Any, Mapping, Protocol, TypeVar, Union


class ArrayLike(Protocol):
    def __getitem__(self, indices: Any, /) -> Any:
        # / is to indicate that the parameter names do not matter
        # https://stackoverflow.com/questions/75420105/python-typing-callback-protocol-and-keyword-arguments
        ...

    def __setitem__(self, indices: Any, value: Any, /) -> None:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def dtype(self) -> Any:
        ...


ArrayType = TypeVar("ArrayType", bound=ArrayLike)
ArrayTree = Union[ArrayType, "ArrayDict[ArrayType]"]
MappingTree = Union[ArrayType, Mapping[str, "MappingTree[ArrayType]"]]

from .array_dict import ArrayDict
from .utils import dict_map

__all__ = [
    "ArrayLike",
    "ArrayType",
    "ArrayTree",
    "ArrayDict",
    "MappingTree",
    "dict_map",
]
