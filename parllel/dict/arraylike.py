from typing import Any, Protocol


class ArrayLike(Protocol):
    def __getitem__(self, key: Any) -> Any: ...

    def __setitem__(self, key: Any, value: Any) -> None: ...

    @property
    def shape(self) -> tuple[int, ...]: ...
