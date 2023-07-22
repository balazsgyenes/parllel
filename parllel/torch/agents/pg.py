from __future__ import annotations

from typing import TypedDict

from torch import Tensor
from typing_extensions import NotRequired

from parllel import MappingTree


class PgPrediction(TypedDict):
    dist_params: MappingTree
    value: NotRequired[Tensor]
