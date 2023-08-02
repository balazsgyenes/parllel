from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from gymnasium import spaces
from torch import Tensor

from parllel.torch.agents.sac_agent import PiModelOutputs, QModelOutputs
from parllel.torch.models import MlpModel


class PiMlpModel(nn.Module):
    """Action distrubition MLP model for SAC agent."""

    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Box,
        hidden_sizes: int | Sequence[int] | None,
        hidden_nonlinearity: str,
    ):
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        assert len(obs_space.shape) == 1
        obs_size = obs_space.shape[0]

        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1
        self._action_size = action_size = action_space.shape[0]

        hidden_nonlinearity = getattr(nn, hidden_nonlinearity)

        self.mlp = MlpModel(
            input_size=obs_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(self, observation: Tensor) -> PiModelOutputs:
        assert len(observation.shape) == 2

        output = self.mlp(observation)
        mean, log_std = output[:, : self._action_size], output[:, self._action_size :]
        return PiModelOutputs(mean=mean, log_std=log_std)


class QMlpModel(nn.Module):
    """Q portion of the model for DDPG, an MLP."""

    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Box,
        hidden_sizes: int | Sequence[int] | None,
        hidden_nonlinearity: str,
    ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        assert len(obs_space.shape) == 1
        obs_size = obs_space.shape[0]

        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1
        action_size = action_space.shape[0]

        hidden_nonlinearity = getattr(nn, hidden_nonlinearity)

        self.mlp = MlpModel(
            input_size=obs_size + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(self, observation: Tensor, action: Tensor) -> QModelOutputs:
        assert len(observation.shape) == 2
        assert len(action.shape) == 2

        q_input = torch.cat([observation, action], dim=1)
        q = self.mlp(q_input).squeeze(-1)
        return QModelOutputs(q_value=q)
