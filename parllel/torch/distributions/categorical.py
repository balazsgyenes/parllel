from typing import Mapping

import torch
from torch import Tensor

from parllel.torch.utils import select_at_indexes

from .discrete import DiscreteMixin
from .distribution import Distribution

DistInfoType = Mapping[str, Tensor]
EPS = 1e-8


class Categorical(DiscreteMixin, Distribution):
    """Multinomial distribution over a discrete domain."""

    def sample(self, dist_info: DistInfoType) -> Tensor:
        """Sample from ``torch.multiomial`` over trailing dimension of
        ``dist_info.prob``."""
        p = dist_info["prob"]
        sample = torch.multinomial(p.view(-1, self.dim), num_samples=1)
        return sample.view(p.shape[:-1]).type(self.dtype)  # Returns indexes.

    def kl(self, old_dist_info: DistInfoType, new_dist_info: DistInfoType) -> Tensor:
        p = old_dist_info["prob"]
        q = new_dist_info["prob"]
        return torch.sum(p * (torch.log(p + EPS) - torch.log(q + EPS)), dim=-1)

    def entropy(self, dist_info: DistInfoType) -> Tensor:
        p = dist_info["prob"]
        return -torch.sum(p * torch.log(p + EPS), dim=-1)

    def log_likelihood(self, indexes: Tensor, /, dist_info: DistInfoType) -> Tensor:
        selected_likelihood = select_at_indexes(indexes, dist_info["prob"])
        return torch.log(selected_likelihood + EPS)

    def likelihood_ratio(
        self,
        indexes: Tensor,
        /,
        old_dist_info: DistInfoType,
        new_dist_info: DistInfoType,
    ) -> Tensor:
        num = select_at_indexes(indexes, new_dist_info["prob"])
        den = select_at_indexes(indexes, old_dist_info["prob"])
        return (num + EPS) / (den + EPS)
