from __future__ import annotations

from typing import Optional

import torch
from gymnasium import spaces
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius

from parllel.torch.agents.categorical import ModelOutputs


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNetPgModel(torch.nn.Module):
    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Discrete,
        hidden_sizes: int | list[int] | None = None,
        hidden_nonlinearity: Optional[str] = None,
    ):
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        obs_shape = obs_space.shape[0]

        assert isinstance(action_space, spaces.Discrete)
        n_actions = action_space.n

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

    def forward(self, data):
        # convert to pytorch geometric batch representation
        pos, ptr = data.pos, data.ptr
        num_nodes = ptr[1:] - ptr[:-1]
        batch = torch.repeat_interleave(
            torch.arange(len(num_nodes), device=num_nodes.device),
            repeats=num_nodes,
        )

        sa0_out = (None, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return ModelOutputs(
            pi=self.pi(x).softmax(dim=-1),
            value=self.value(x).squeeze(-1),
        )
