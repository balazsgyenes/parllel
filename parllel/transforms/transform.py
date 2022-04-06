from abc import ABC, abstractmethod
from typing import Sequence, Tuple

from parllel.buffers import Samples


class Transform(ABC):
    pass


class BatchTransform(Transform):
    @abstractmethod
    def __call__(self, batch_samples: Samples) -> Samples:
        raise NotImplementedError


class StepTransform(Transform):
    @abstractmethod
    def __call__(self, batch_samples: Samples, t: int) -> Samples:
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: Sequence[Transform]) -> None:
        if not (all(isinstance(transform, BatchTransform) for transform in transforms)
             or all(isinstance(transform, StepTransform) for transform in transforms)):
             raise ValueError("Not allowed to mix StepTransforms and BatchTransforms")

        self.transforms: Tuple[Transform] = tuple(transforms)

    def __call__(self, batch_samples: Samples, *args, **kwargs) -> Samples:
        for transform in self.transforms:
            batch_samples = transform(batch_samples, *args, **kwargs)
        return batch_samples
