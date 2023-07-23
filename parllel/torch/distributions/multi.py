from typing import Mapping

import torch
from torch import Tensor

from parllel import ArrayOrMapping

from .distribution import Distribution

ActionType = Mapping[str, ArrayOrMapping[Tensor]]
DistParamsType = Mapping[str, Mapping[str, ArrayOrMapping[Tensor]]]


class MultiDistribution(Distribution):
    def __init__(self, distributions: dict[str, Distribution]) -> None:
        self._distributions = distributions

    def sample(self, dist_params: DistParamsType) -> ActionType:
        raise NotImplementedError

    def log_likelihood(self, action: ActionType, dist_params: DistParamsType) -> Tensor:
        likelihoods = [
            distribution.log_likelihood(action[name], dist_params[name])
            for name, distribution in self._distributions.items()
        ]
        likelihood = torch.stack(likelihoods, dim=-1)
        return likelihood

    def likelihood_ratio(
        self,
        action: ActionType,
        old_dist_params: DistParamsType,
        new_dist_params: DistParamsType,
    ) -> Tensor:
        ratios = [
            distribution.likelihood_ratio(
                action[name],
                old_dist_params=old_dist_params[name],
                new_dist_params=new_dist_params[name],
            )
            for name, distribution in self._distributions.items()
        ]
        return torch.stack(ratios, dim=-1)

    def entropy(self, dist_params: DistParamsType) -> Tensor:
        entropies = [
            distribution.entropy(dist_params=dist_params[name])
            for name, distribution in self._distributions.items()
        ]
        return torch.stack(entropies, dim=-1)
