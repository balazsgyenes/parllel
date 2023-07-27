from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.nn import PointNetConv, fps, global_max_pool, radius

from parllel import ArrayDict


class SAModule(torch.nn.Module):
    def __init__(self, ratio: float, r: float, nn: torch.nn.Module) -> None:
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(
        self,
        x: Tensor | None,
        pos: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
    def __init__(self, nn: torch.nn.Module) -> None:
        super().__init__()
        self.nn = nn

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def dict_to_batched_data(
    array_dict: ArrayDict[Tensor],
) -> tuple[Tensor, Tensor]:
    pos, ptr = array_dict["pos"], array_dict["ptr"]
    num_nodes = ptr[1:] - ptr[:-1]
    batch = torch.repeat_interleave(
        torch.arange(len(num_nodes), device=num_nodes.device),
        repeats=num_nodes,
    )

    return pos, batch
