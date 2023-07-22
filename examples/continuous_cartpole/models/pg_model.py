from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from gymnasium import spaces
from torch import Tensor

from parllel.torch.agents.gaussian import DistParams, ModelOutputs
from parllel.torch.models import MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims


class GaussianCartPoleFfPgModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Discrete,
        hidden_sizes: int | Sequence[int] | None,
        hidden_nonlinearity: str,
        mu_nonlinearity: str | None,
        init_log_std: float,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        obs_shape = obs_space.shape[0]

        assert isinstance(action_space, spaces.Box)
        action_size = action_space.shape[0]

        hidden_nonlinearity = getattr(nn, hidden_nonlinearity)

        mu_mlp = MlpModel(
            input_size=obs_shape,
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
            input_size=obs_shape,
            hidden_sizes=hidden_sizes,
            output_size=1,
            hidden_nonlinearity=hidden_nonlinearity,
        )
        self.log_std = nn.Parameter(torch.full((action_size,), init_log_std))

    def forward(self, observation: Tensor) -> ModelOutputs:
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)

        obs_flat = observation.view(T * B, -1)
        mu = self.mu(obs_flat)
        value = self.value(obs_flat).squeeze(-1)
        log_std = self.log_std.repeat(T * B, 1)

        mu, value, log_std = restore_leading_dims((mu, value, log_std), lead_dim, T, B)

        return ModelOutputs(
            dist_params=DistParams(mean=mu, log_std=log_std),
            value=value,
        )
