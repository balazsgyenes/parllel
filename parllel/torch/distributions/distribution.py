from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from parllel.torch.utils import valid_mean


class Distribution(ABC): 
    """Base distribution class.  Not all subclasses will implement all
    methods."""

    def to_device(self, device: torch.device) -> None:
        pass

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def sample(self, dist_info: Any) -> Any:
        """Generate random sample(s) from distribution informations."""
        raise NotImplementedError

    def kl(self, old_dist_info: Any, new_dist_info: Any) -> torch.Tensor:
        """
        Compute the KL divergence of two distributions at each datum; should
        maintain leading dimensions (e.g. [T,B]).
        """
        raise NotImplementedError

    def log_likelihood(self, x: Any, /, dist_info: Any) -> torch.Tensor:
        """
        Compute log-likelihood of samples ``x`` at distributions described in
        ``dist_info`` (i.e. can have same leading dimensions [T, B]).
        """
        raise NotImplementedError

    def likelihood_ratio(self, x: Any, /, old_dist_info: Any, new_dist_info: Any) -> torch.Tensor:
        """
        Compute likelihood ratio of samples ``x`` at new distributions over
        old distributions (usually ``new_dist_info`` is variable for
        differentiation); should maintain leading dimensions.
        """
        raise NotImplementedError

    def entropy(self, dist_info: Any) -> torch.Tensor:
        """
        Compute entropy of distributions contained in ``dist_info``; should
        maintain any leading dimensions.
        """
        raise NotImplementedError

    def mean_kl(self,
        old_dist_info: Any,
        new_dist_info: Any,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the mean KL divergence over a data batch, possible ignoring
        data marked as invalid.
        """
        raise valid_mean(self.kl(old_dist_info, new_dist_info), valid)

    def perplexity(self, dist_info: Any) -> torch.Tensor:
        """Exponential of the entropy, maybe useful for logging."""
        return torch.exp(self.entropy(dist_info))

    def mean_entropy(self,
        dist_info: Any,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the mean entropy over a data batch, possible ignoring
        data marked as invalid.
        """
        return valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self,
        dist_info: Any,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the mean perplexity over a data batch, possible ignoring
        data marked as invalid.
        """
        return valid_mean(self.perplexity(dist_info), valid)
