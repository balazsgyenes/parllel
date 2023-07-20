from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from parllel.torch.distributions.distribution import DistInfoType


@dataclass(frozen=True)
class PgPrediction:
    dist_info: DistInfoType
    value: Tensor | None = None
