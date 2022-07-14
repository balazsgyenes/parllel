from typing import Union

import torch

from parllel.torch.distributions.gaussian import (Gaussian, DistInfo,
    DistInfoStd)


EPS = 1e-8


class SquashedGaussian(Gaussian):
    def __init__(
            self,
            scale: Union[float, torch.Tensor],
            *args,
            **kwargs,
            ):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def kl(self, old_dist_info, new_dist_info):
        raise NotImplementedError

    def entropy(self, dist_info):
        raise NotImplementedError

    def log_likelihood(self, x, dist_info):
        # TODO: an implementation for this exists in SB3, even though it's not
        # very efficient
        raise NotImplementedError

    def likelihood_ratio(self, x, old_dist_info, new_dist_info):
        raise NotImplementedError

    def sample_loglikelihood(self, dist_info):
        """Special method for use with SAC algorithm, which efficiently
        computes a new sampled action and its log-likelihood for optimization
        use. The log-likelihood requires the unsquashed sample, so instead of
        squashing and unsquashing, the parent methods are called and the sample
        is squashed afterwards. Then, the log-likelihood is corrected to take
        the squashing into account. The interested reader should refer to
        Appendix C of the SAC paper by Haarnoja et al. (2018).
        
        TODO: verify correction factor with scale not equal 1 (not included in
            paper).
        """
        # sample unsquashed Gaussian distribution
        sample = super().sample(dist_info)

        # compute log likelihood of unsquashed sample
        logli = super().log_likelihood(sample, dist_info)

        # squash sample and add correction to log likelihood
        tanh_x = torch.tanh(sample)
        logli -= torch.sum(
            torch.log(self.scale * (1 - tanh_x ** 2) + EPS),
            dim=-1,
        )
        sample = self.scale * tanh_x
        
        return sample, logli

    def sample(self, dist_info):
        """
        Generate random samples using ``torch.normal``, from
        ``dist_info.mean``. Uses ``self.std`` unless it is ``None``, then uses
        ``dist_info.log_std``.
        """
        sample = super().sample(dist_info)
        sample = self.scale * torch.tanh(sample)
        return sample

    def set_scale(self, scale):
        """Input multiplicative factor for ``scale * tanh(sample)`` (usually
        will be 1)."""
        self.scale = scale
