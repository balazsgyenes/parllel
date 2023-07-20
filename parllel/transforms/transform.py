from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

from parllel import Array, ArrayDict


class Transform(ABC):
    pass


class BatchTransform(Transform):
    @abstractmethod
    def __call__(self, batch_samples: ArrayDict[Array]) -> ArrayDict[Array]:
        raise NotImplementedError


class StepTransform(Transform):
    @abstractmethod
    def __call__(self, batch_samples: ArrayDict[Array], t: int) -> ArrayDict[Array]:
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: Sequence[Transform]) -> None:
        if not (all(isinstance(transform, BatchTransform) for transform in transforms)
             or all(isinstance(transform, StepTransform) for transform in transforms)):
             raise ValueError("Not allowed to mix StepTransforms and BatchTransforms")

        self.transforms = tuple(transforms)

    def __call__(self, batch_samples: ArrayDict[Array], *args, **kwargs) -> ArrayDict[Array]:
        for transform in self.transforms:
            batch_samples = transform(batch_samples, *args, **kwargs)
        return batch_samples
