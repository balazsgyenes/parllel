from __future__ import annotations

from abc import ABC, abstractmethod

from parllel import Array, ArrayDict


class Transform(ABC):
    @abstractmethod
    def __call__(self, batch_samples: ArrayDict[Array]) -> ArrayDict[Array]:
        raise NotImplementedError
