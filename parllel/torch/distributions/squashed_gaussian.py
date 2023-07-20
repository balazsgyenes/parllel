from typing import SupportsFloat

import torch
from torch import Tensor

from parllel.torch.distributions.gaussian import DistInfoType, Gaussian

EPS = 1e-8


class SquashedGaussian(Gaussian):
    def __init__(
        self,
        scale: SupportsFloat,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def sample(self, dist_info: DistInfoType) -> Tensor:
        """
        Generate random samples using ``torch.normal``, from
        ``dist_info.mean``. Uses ``self.std`` unless it is ``None``, then uses
        ``dist_info.log_std``.
        """
        sample = super().sample(dist_info)
        sample = self.scale * torch.tanh(sample)
        return sample

    def sample_loglikelihood(self, dist_info: DistInfoType) -> tuple[Tensor, Tensor]:
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
            torch.log(self.scale * (1 - tanh_x**2) + EPS),
            dim=-1,
        )
        sample = self.scale * tanh_x

        return sample, logli

    def kl(self, old_dist_info: DistInfoType, new_dist_info: DistInfoType) -> Tensor:
        raise NotImplementedError

    def entropy(self, dist_info: DistInfoType) -> Tensor:
        raise NotImplementedError

    def log_likelihood(self, indexes: Tensor, /, dist_info: DistInfoType) -> Tensor:
        # TODO: an implementation for this exists in SB3, even though it's not
        # very efficient
        raise NotImplementedError

    def likelihood_ratio(
        self,
        indexes: Tensor,
        /,
        old_dist_info: DistInfoType,
        new_dist_info: DistInfoType,
    ) -> Tensor:
        raise NotImplementedError

    def set_scale(self, scale: SupportsFloat) -> None:
        """Input multiplicative factor for ``scale * tanh(sample)`` (usually
        will be 1)."""
        self.scale = scale
