from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import torch
from torch import Tensor

from parllel import ArrayOrMapping
from parllel.torch.utils import valid_mean

ActionType = ArrayOrMapping[Tensor]
DistParamsType = Mapping[str, ArrayOrMapping]


class Distribution(ABC):
    """Base distribution class.  Not all subclasses will implement all
    methods."""

    def to_device(self, device: torch.device) -> None:
        pass

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def sample(self, dist_params: DistParamsType) -> ActionType:
        """Generate random sample(s) from distribution informations."""
        raise NotImplementedError

    def kl(
        self,
        old_dist_params: DistParamsType,
        new_dist_params: DistParamsType,
    ) -> Tensor:
        """
        Compute the KL divergence of two distributions at each datum; should
        maintain leading dimensions (e.g. [T,B]).
        """
        raise NotImplementedError

    def log_likelihood(
        self,
        x: ActionType,
        /,
        dist_params: DistParamsType,
    ) -> Tensor:
        """
        Compute log-likelihood of samples ``x`` at distributions described in
        ``dist_params`` (i.e. can have same leading dimensions [T, B]).
        """
        raise NotImplementedError

    def likelihood_ratio(
        self,
        x: ActionType,
        /,
        old_dist_params: DistParamsType,
        new_dist_params: DistParamsType,
    ) -> Tensor:
        """
        Compute likelihood ratio of samples ``x`` at new distributions over
        old distributions (usually ``new_dist_params`` is variable for
        differentiation); should maintain leading dimensions.
        """
        raise NotImplementedError

    def entropy(self, dist_params: DistParamsType) -> Tensor:
        """
        Compute entropy of distributions contained in ``dist_params``; should
        maintain any leading dimensions.
        """
        raise NotImplementedError

    def mean_kl(
        self,
        old_dist_params: DistParamsType,
        new_dist_params: DistParamsType,
        valid: Tensor | None = None,
    ) -> Tensor:
        """Compute the mean KL divergence over a data batch, possible ignoring
        data marked as invalid.
        """
        return valid_mean(self.kl(old_dist_params, new_dist_params), valid)

    def perplexity(self, dist_params: DistParamsType) -> Tensor:
        """Exponential of the entropy, maybe useful for logging."""
        return torch.exp(self.entropy(dist_params))

    def mean_entropy(
        self,
        dist_params: DistParamsType,
        valid: Tensor | None = None,
    ) -> Tensor:
        """Compute the mean entropy over a data batch, possible ignoring
        data marked as invalid.
        """
        return valid_mean(self.entropy(dist_params), valid)

    def mean_perplexity(
        self,
        dist_params: DistParamsType,
        valid: Tensor | None = None,
    ) -> Tensor:
        """Compute the mean perplexity over a data batch, possible ignoring
        data marked as invalid.
        """
        return valid_mean(self.perplexity(dist_params), valid)

    def eval(self):
        pass

    def train(self):
        pass
