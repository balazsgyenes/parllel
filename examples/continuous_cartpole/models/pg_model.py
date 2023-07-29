from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from gymnasium import spaces
from torch import Tensor

from parllel.torch.agents.gaussian import DistParams, ModelOutputs
from parllel.torch.models import MlpModel


class GaussianCartPoleFfPgModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Box,
        hidden_sizes: int | Sequence[int] | None,
        hidden_nonlinearity: str,
        mu_nonlinearity: str | None,
        init_log_std: float,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        assert len(obs_space.shape) == 1
        obs_size = obs_space.shape[0]

        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1
        action_size = action_space.shape[0]

        hidden_nonlinearity = getattr(nn, hidden_nonlinearity)

        mu_mlp = MlpModel(
            input_size=obs_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size,
            hidden_nonlinearity=hidden_nonlinearity,
        )
        if mu_nonlinearity is not None:
            mu_nonlinearity = getattr(nn, mu_nonlinearity)
            self.mu = nn.Sequential(mu_mlp, mu_nonlinearity())
        else:
            self.mu = mu_mlp

        self.value = MlpModel(
            input_size=obs_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            hidden_nonlinearity=hidden_nonlinearity,
        )
        self.log_std = nn.Parameter(torch.full((action_size,), init_log_std))

    def forward(self, observation: Tensor) -> ModelOutputs:
        assert len(observation.shape) == 2

        mean = self.mu(observation)
        value = self.value(observation).squeeze(-1)
        log_std = self.log_std.repeat(mean.shape[0], 1)
        return ModelOutputs(
            dist_params=DistParams(mean=mean, log_std=log_std),
            value=value,
        )
