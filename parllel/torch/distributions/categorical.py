from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from parllel.torch.utils import select_at_indexes

from .discrete import DiscreteMixin
from .distribution import Distribution

DistParamsTree = Mapping[str, Tensor]
EPS = 1e-8


@dataclass(frozen=True)
class DistParams:
    probs: Tensor


class Categorical(DiscreteMixin, Distribution[DistParams]):
    """Multinomial distribution over a discrete domain."""

    def sample(self, dist_params: DistParams) -> Tensor:
        """Sample from ``torch.multiomial`` over trailing dimension of
        ``dist_params.probs``."""
        p = dist_params.probs
        sample = torch.multinomial(p.view(-1, self.dim), num_samples=1)
        return sample.view(p.shape[:-1]).type(self.dtype)  # Returns indexes.

    def kl(self, old_dist_params: DistParamsTree, new_dist_params: DistParamsTree) -> Tensor:
        p = old_dist_params["probs"]
        q = new_dist_params["probs"]
        return torch.sum(p * (torch.log(p + EPS) - torch.log(q + EPS)), dim=-1)

    def log_likelihood(self, indexes: Tensor, /, dist_params: DistParamsTree) -> Tensor:
        selected_likelihood = select_at_indexes(indexes, dist_params["probs"])
        return torch.log(selected_likelihood + EPS)

    def likelihood_ratio(
        self,
        indexes: Tensor,
        /,
        old_dist_params: DistParamsTree,
        new_dist_params: DistParamsTree,
    ) -> Tensor:
        num = select_at_indexes(indexes, new_dist_params["probs"])
        den = select_at_indexes(indexes, old_dist_params["probs"])
        return (num + EPS) / (den + EPS)

    def entropy(self, dist_params: DistParamsTree) -> Tensor:
        p = dist_params["probs"]
        return -torch.sum(p * torch.log(p + EPS), dim=-1)
