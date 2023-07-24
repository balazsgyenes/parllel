from __future__ import annotations

import torch
from torch import Tensor

from parllel.torch.utils import to_onehot, from_onehot


class DiscreteMixin:
    """Conversions to and from one-hot."""

    def __init__(
        self,
        dim: int,
        dtype: torch.dtype = torch.long,
        onehot_dtype: torch.dtype = torch.float,
    ):
        self._dim = dim
        self.dtype = dtype
        self.onehot_dtype = onehot_dtype

    @property
    def dim(self) -> int:
        return self._dim

    def to_onehot(self, indexes: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        """Convert from integer indexes to one-hot, preserving leading dimensions."""
        return to_onehot(indexes, self._dim, dtype=dtype or self.onehot_dtype)

    def from_onehot(self, onehot: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        """Convert from one-hot to integer indexes, preserving leading dimensions."""
        return from_onehot(onehot, dtype=dtype or self.dtype)
