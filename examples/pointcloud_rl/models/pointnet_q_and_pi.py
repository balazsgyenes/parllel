from __future__ import annotations

import torch
from gymnasium import spaces
from torch import Tensor
from torch_geometric.nn import MLP

from parllel.torch.agents.sac_agent import PiModelOutputs, QModelOutputs


class PointNetPiModel(torch.nn.Module):
    def __init__(
        self,
        encoding_size: int,
        action_space: spaces.Discrete,
        hidden_sizes: int | list[int] | None = None,
        hidden_nonlinearity: str | None = None,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1
        self._action_size = action_size = action_space.shape[0]

        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        else:
            try:
                hidden_sizes = list(hidden_sizes)
            except TypeError:
                hidden_sizes = [int(hidden_sizes)]

        self.mlp = MLP(
            [encoding_size] + hidden_sizes + [action_size * 2],
            norm=None,
            act=hidden_nonlinearity,
        )

    def forward(self, encoding: Tensor) -> PiModelOutputs:
        output = self.mlp(encoding)
        mean, log_std = output[:, : self._action_size], output[:, self._action_size :]
        return PiModelOutputs(mean=mean, log_std=log_std)


class PointNetQModel(torch.nn.Module):
    def __init__(
        self,
        encoding_size: int,
        action_space: spaces.Discrete,
        hidden_sizes: int | list[int] | None = None,
        hidden_nonlinearity: str | None = None,
    ):
        super().__init__()
        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1
        action_size = action_space.shape[0]

        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        else:
            try:
                hidden_sizes = list(hidden_sizes)
            except TypeError:
                hidden_sizes = [int(hidden_sizes)]

        self.mlp = MLP(
            # action is concatenated onto observation
            [encoding_size + action_size] + hidden_sizes + [1],
            norm=None,
            act=hidden_nonlinearity,
        )

    def forward(self, encoding: Tensor, action: Tensor) -> QModelOutputs:
        q_input = torch.cat([encoding, action], dim=1)
        q = self.mlp(q_input).squeeze(-1)
        return QModelOutputs(q_value=q)
