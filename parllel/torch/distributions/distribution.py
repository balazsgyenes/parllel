from abc import ABC, abstractmethod

import torch

from parllel.torch.utils import valid_mean


class Distribution(ABC): 
    """Base distribution class.  Not all subclasses will implement all
    methods."""

    def to_device(self, device: torch.device):
        # TODO: implement, e.g. for Gaussian with fixed std
        pass

    @property
    def dim(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self, dist_info):
        """Generate random sample(s) from distribution informations."""
        raise NotImplementedError

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two distributions at each datum; should
        maintain leading dimensions (e.g. [T,B]).
        """
        raise NotImplementedError

    def log_likelihood(self, x, /, dist_info):
        """
        Compute log-likelihood of samples ``x`` at distributions described in
        ``dist_info`` (i.e. can have same leading dimensions [T, B]).
        """
        raise NotImplementedError

    def likelihood_ratio(self, x, /, old_dist_info, new_dist_info):
        """
        Compute likelihood ratio of samples ``x`` at new distributions over
        old distributions (usually ``new_dist_info`` is variable for
        differentiation); should maintain leading dimensions.
        """
        raise NotImplementedError

    def entropy(self, dist_info):
        """
        Compute entropy of distributions contained in ``dist_info``; should
        maintain any leading dimensions.
        """
        raise NotImplementedError

    def mean_kl(self, old_dist_info, new_dist_info, valid = None):
        """Compute the mean KL divergence over a data batch, possible ignoring
        data marked as invalid.
        """
        raise valid_mean(self.kl(old_dist_info, new_dist_info), valid)

    def perplexity(self, dist_info):
        """Exponential of the entropy, maybe useful for logging."""
        return torch.exp(self.entropy(dist_info))

    def mean_entropy(self, dist_info, valid = None):
        """Compute the mean entropy over a data batch, possible ignoring
        data marked as invalid.
        """
        return valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self, dist_info, valid = None):
        """Compute the mean perplexity over a data batch, possible ignoring
        data marked as invalid.
        """
        return valid_mean(self.perplexity(dist_info), valid)
