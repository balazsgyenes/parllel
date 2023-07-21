from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from parllel.torch.distributions.distribution import DistParamsTree


@dataclass(frozen=True)
class PgPrediction:
    dist_params: DistParamsTree
    value: Tensor | None = None
