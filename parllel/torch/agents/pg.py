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
    # must specify Distribution, otherwise its type is Unknown
    TorchAgent[torch.nn.Module, Distribution],
    # must reinherit from Generic so that child classes can specialize further
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
