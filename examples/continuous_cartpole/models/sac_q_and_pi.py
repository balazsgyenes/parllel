from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch import Tensor

from parllel.torch.agents.sac_agent import PiModelOutputs, QModelOutputs
from parllel.torch.models import MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims


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
        self._obs_ndim = len(obs_space.shape)

        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1
        self._action_size = action_size = action_space.shape[0]

        hidden_nonlinearity = getattr(nn, hidden_nonlinearity)

        self.mlp = MlpModel(
            input_size=int(np.prod(obs_space.shape)),
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(self, observation: Tensor) -> PiModelOutputs:
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        output = self.mlp(observation.view(T * B, -1))
        mean, log_std = output[:, : self._action_size], output[:, self._action_size :]
        mean, log_std = restore_leading_dims((mean, log_std), lead_dim, T, B)
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
        self._obs_ndim = len(obs_space.shape)

        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1
        action_size = action_space.shape[0]

        hidden_nonlinearity = getattr(nn, hidden_nonlinearity)

        self.mlp = MlpModel(
            input_size=int(np.prod(obs_space.shape)) + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(self, observation: Tensor, action: Tensor) -> QModelOutputs:
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        q_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1
        )
        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return QModelOutputs(q_value=q)
