from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from torch import Tensor

from parllel.torch.agents.categorical import DistParams, ModelOutputs
from parllel.torch.utils import to_onehot


class LookupTablePgModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Discrete,
        action_space: spaces.Discrete,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, spaces.Discrete)
        obs_size = self.obs_size = int(obs_space.n)

        assert isinstance(action_space, spaces.Discrete)
        n_actions = int(action_space.n)

        self.pi = nn.Linear(obs_size, n_actions)
        self.value = nn.Linear(obs_size, 1)

    def forward(self, observation: Tensor) -> ModelOutputs:
        observation = to_onehot(observation, num=self.obs_size, dtype=torch.float32)
        probs = self.pi(observation).softmax(dim=-1)
        value = self.value(observation).squeeze(-1)
        return ModelOutputs(dist_params=DistParams(probs=probs), value=value)
