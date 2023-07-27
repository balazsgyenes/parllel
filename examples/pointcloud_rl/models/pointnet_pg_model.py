from __future__ import annotations

import torch
from gymnasium import spaces
from pointcloud import PointCloudSpace
from torch import Tensor
from torch_geometric.nn import MLP

from parllel import ArrayDict
from parllel.torch.agents.categorical import DistParams, ModelOutputs

from .modules import GlobalSAModule, SAModule, dict_to_batched_data


class PointNetPgModel(torch.nn.Module):
    def __init__(
        self,
        obs_space: PointCloudSpace,
        action_space: spaces.Discrete,
        hidden_sizes: int | list[int] | None = None,
        hidden_nonlinearity: str | None = None,
    ):
        super().__init__()
        assert isinstance(obs_space, PointCloudSpace)
        obs_shape = obs_space.shape[0]

        assert isinstance(action_space, spaces.Discrete)
        n_actions = int(action_space.n)

        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 256]
        else:
            try:
                hidden_sizes = list(hidden_sizes)
            except TypeError:
                hidden_sizes = [int(hidden_sizes)]

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([obs_shape, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + obs_shape, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + obs_shape, 256, 512, 1024]))

        self.pi = MLP(hidden_sizes + [n_actions], norm=None, act=hidden_nonlinearity)
        self.value = MLP(hidden_sizes + [1], norm=None, act=hidden_nonlinearity)

    def forward(self, data: ArrayDict[Tensor]) -> ModelOutputs:
        # convert to pytorch geometric batch representation
        pos, batch = dict_to_batched_data(data)

        sa0_out = (None, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        probs = self.pi(x).softmax(dim=-1)
        value = self.value(x).squeeze(-1)

        return ModelOutputs(dist_params=DistParams(probs=probs), value=value)
