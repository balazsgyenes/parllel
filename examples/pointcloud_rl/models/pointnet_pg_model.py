from __future__ import annotations

import torch
from gymnasium import spaces
from pointcloud import PointCloudSpace
from torch import Tensor
from torch_geometric.nn import MLP

from parllel import ArrayDict
from parllel.torch.agents.categorical import DistParams, ModelOutputs

from .modules import PointNetEncoder


class PointNetPgModel(torch.nn.Module):
    def __init__(
        self,
        obs_space: PointCloudSpace,
        action_space: spaces.Discrete,
        encoding_size: int | None = None,
        hidden_sizes: int | list[int] | None = None,
        hidden_nonlinearity: str | None = None,
    ):
        super().__init__()
        self.encoder = PointNetEncoder(
            obs_space=obs_space,
            encoding_size=encoding_size,
        )
        encoding_size = self.encoder.encoding_size

        assert isinstance(action_space, spaces.Discrete)
        n_actions = int(action_space.n)

        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        else:
            try:
                hidden_sizes = list(hidden_sizes)
            except TypeError:
                hidden_sizes = [int(hidden_sizes)]

        self.pi = MLP(
            [encoding_size] + hidden_sizes + [n_actions],
            norm=None,
            act=hidden_nonlinearity,
        )
        self.value = MLP(
            [encoding_size] + hidden_sizes + [1],
            norm=None,
            act=hidden_nonlinearity,
        )

    def forward(self, data: ArrayDict[Tensor]) -> ModelOutputs:
        encoding = self.encoder(data)
        probs = self.pi(encoding).softmax(dim=-1)
        value = self.value(encoding).squeeze(-1)
        return ModelOutputs(dist_params=DistParams(probs=probs), value=value)
