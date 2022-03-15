from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, Sequence, Tuple

from parllel.buffers import Buffer


class Transform(ABC):
    @abstractmethod
    def __call__(self, samples: Buffer) -> Buffer:
        raise NotImplementedError

class Compose(Transform):
    def __init__(self, transforms: Sequence[Callable[[Buffer], Buffer]]) -> None:
        self.transforms: Tuple[Callable[[Buffer], Buffer]] = tuple(transforms)

    def __call__(self, samples: Buffer) -> Buffer:
        return reduce(lambda buf, f: f(buf), self.transforms, samples)
