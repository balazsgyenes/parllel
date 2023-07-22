from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypedDict

import torch
from torch import Tensor
from typing_extensions import NotRequired

from parllel import ArrayDict, ArrayTree, MappingTree
from parllel.torch.distributions.distribution import Distribution

from .agent import DistType, ModelType, TorchAgent


class PgPrediction(TypedDict):
    dist_params: MappingTree
    value: NotRequired[Tensor]


class PgAgent(
    TorchAgent[torch.nn.Module, Distribution],
    # must reinherit from Generic to pass it on to child classes
    Generic[ModelType, DistType],
):
    @abstractmethod
    def predict(
        self,
        observation: ArrayTree[Tensor],
        agent_info: ArrayDict[Tensor],
        init_rnn_state: ArrayTree[Tensor] | None,
    ) -> PgPrediction:
        pass
