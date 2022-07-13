from typing import Dict

import torch

from .distribution import Distribution


class MultiDistribution(Distribution):
    def __init__(self, distributions: Dict[str, Distribution]) -> None:
        self._distributions = distributions
        
    def sample(self, dist_info):
        raise NotImplementedError

    def log_likelihood(self, action, dist_info):
        likelihoods = list()
        for name, distribution in self._distributions.items():
            likelihood = distribution.log_likelihood(
                indexes=getattr(action, name),
                dist_info=getattr(dist_info, name),
            )
            likelihoods.append(likelihood)

        likelihood = torch.stack(likelihoods, dim=-1)
        return likelihood

    def likelihood_ratio(self, action, old_dist_info, new_dist_info):
        ratios = list()
        for name, distribution in self._distributions.items():
            ratio = distribution.likelihood_ratio(
                getattr(action, name),
                old_dist_info=getattr(old_dist_info, name),
                new_dist_info=getattr(new_dist_info, name),
            )
            ratios.append(ratio)

        ratio = torch.stack(ratios, dim=-1)
        return ratio

    def entropy(self, dist_info):
        entropies = list()
        for name, distribution in self._distributions.items():
            ratio = distribution.entropy(
                dist_info=getattr(dist_info, name),
            )
            entropies.append(ratio)

        ratio = torch.stack(entropies, dim=-1)
        return ratio
