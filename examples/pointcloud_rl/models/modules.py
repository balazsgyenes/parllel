from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius

import parllel.logger as logger
from parllel import ArrayDict

# isort: split
from pointcloud import PointCloudSpace


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
        if len(torch.unique(batch)) != batch.max() + 1:
            logger.error(
                "Program is about to crash in fps function due to empty pointcloud."
            )
        idx = fps(pos, batch, ratio=self.ratio)
        if len(torch.unique(batch)) != batch.max() + 1:
            logger.error("If this has printed, the previous message was a lie.")
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


class PointNetEncoder(torch.nn.Module):
    def __init__(
        self,
        obs_space: PointCloudSpace,
        encoding_size: int | None = None,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, PointCloudSpace)
        obs_shape = obs_space.shape[0]

        self._encoding_size = encoding_size or 1024

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(
            ratio=0.5,
            r=0.2,
            nn=MLP([obs_shape, 64, 64, 128]),
        )
        self.sa2_module = SAModule(
            ratio=0.25,
            r=0.4,
            nn=MLP([128 + obs_shape, 128, 128, 256]),
        )
        self.sa3_module = GlobalSAModule(
            nn=MLP([256 + obs_shape, 256, 512, self._encoding_size]),
        )

    def forward(self, data: ArrayDict[Tensor]) -> Tensor:
        # convert to pytorch geometric batch representation
        pos, batch = dict_to_batched_data(data)

        sa0_out = (None, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return x

    @property
    def encoding_size(self) -> int:
        return self._encoding_size


def dict_to_batched_data(
    array_dict: ArrayDict[Tensor],
) -> tuple[Tensor, Tensor]:
    pos, ptr = array_dict["pos"], array_dict["ptr"]
    num_nodes = ptr[1:] - ptr[:-1]

    if (num_nodes == 0).any():
        empty_indices = (num_nodes == 0).nonzero(as_tuple=True)[0].tolist()
        logger.warn(
            f"The following point clouds in this batch are empty: {empty_indices}. This will cause a floating point error in the fps function, so they will be removed from the batch. However, this will probably still cause an error elsewhere."
        )
        num_nodes = num_nodes[num_nodes != 0]

    batch = torch.repeat_interleave(
        torch.arange(len(num_nodes), device=num_nodes.device),
        repeats=num_nodes,
    )

    return pos, batch
