from __future__ import annotations

from abc import abstractmethod
from typing import TypedDict

from torch import Tensor
from typing_extensions import NotRequired

from parllel import ArrayDict, ArrayTree, ArrayOrMapping

from .agent import TorchAgent


class PgPrediction(TypedDict):
    dist_params: ArrayOrMapping[Tensor]
    value: NotRequired[Tensor]


class PgAgent(TorchAgent):
    @abstractmethod
    def predict(
        self,
        observation: ArrayTree[Tensor],
        agent_info: ArrayDict[Tensor],
        initial_rnn_state: ArrayTree[Tensor] | None,
    ) -> PgPrediction:
        pass
