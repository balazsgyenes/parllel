from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from gymnasium import spaces
from torch import Tensor

from parllel.torch.agents.categorical import DistParams, ModelOutputs
from parllel.torch.models import MlpModel


class CartPoleFfPgModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Discrete,
        hidden_sizes: int | Sequence[int] | None,
        hidden_nonlinearity: str,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        assert len(obs_space.shape) == 1
        obs_size = obs_space.shape[0]

        assert isinstance(action_space, spaces.Discrete)
        n_actions = int(action_space.n)

        hidden_nonlinearity = getattr(nn, hidden_nonlinearity)

        self.pi = MlpModel(
            input_size=obs_size,
            hidden_sizes=hidden_sizes,
            output_size=n_actions,
            hidden_nonlinearity=hidden_nonlinearity,
        )

        self.value = MlpModel(
            input_size=obs_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(self, observation: Tensor) -> ModelOutputs:
        assert len(observation.shape) == 2

        probs = self.pi(observation).softmax(dim=-1)
        value = self.value(observation).squeeze(-1)
        return ModelOutputs(dist_params=DistParams(probs=probs), value=value)
