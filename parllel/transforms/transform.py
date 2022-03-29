from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

from parllel.samplers import Samples


class Transform(ABC):
    @abstractmethod
    def __call__(self, batch_samples: Samples, t: Optional[int] = None) -> Samples:
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms: Tuple[Transform] = tuple(transforms)

    def __call__(self, batch_samples: Samples, t: Optional[int] = None) -> Samples:
        if t is None:
            for transform in self.transforms:
                batch_samples = transform(batch_samples)
        else:
            for transform in self.transforms:
                batch_samples = transform(batch_samples, t)
        return batch_samples
