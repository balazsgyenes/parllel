from abc import ABC, abstractmethod
from functools import reduce
from typing import Sequence, Tuple

from parllel.buffers import Buffer


class Transform(ABC):
    @abstractmethod
    def __call__(self, samples: Buffer) -> Buffer:
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms: Tuple[Transform] = tuple(transforms)

    def __call__(self, samples: Buffer) -> Buffer:
        return reduce(lambda buffer, transform: transform(buffer),
                      self.transforms, samples)
