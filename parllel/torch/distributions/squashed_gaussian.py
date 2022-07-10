from typing import Union

import torch

from parllel.torch.distributions.gaussian import Gaussian


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
        raise NotImplementedError

    def sample_loglikelihood(self, dist_info):
        """
        Special method for use with SAC algorithm, which returns a new sampled 
        action and its log-likelihood for training use.  Temporarily turns OFF
        squashing, so that log_likelihood can be computed on non-squashed sample,
        and then restores squashing and applies it to the sample before output.
        """
        sample = super().sample(dist_info)
        logli = super().log_likelihood(sample, dist_info)
        # TODO: where does this factor come from?
        logli -= torch.sum(
            torch.log(self.scale * (1 - torch.tanh(x) ** 2) + EPS),
            dim=-1,
        )
        sample = self.scale * torch.tanh(sample)
        return sample, logli

    # def sample_loglikelihood(self, dist_info):
    #     """Use in SAC with squash correction, since log_likelihood() expects raw_action."""
    #     mean = dist_info.mean
    #     log_std = dist_info.log_std
    #     if self.min_log_std is not None or self.max_log_std is not None:
    #         log_std = torch.clamp(log_std, min=self.min_log_std,
    #             max=self.max_log_std)
    #     std = torch.exp(log_std)
    #     normal = torch.distributions.Normal(mean, std)
    #     sample = normal.rsample()
    #     logli = normal.log_prob(sample)
    #     if self.scale is not None:
    #         sample = self.scale * torch.tanh(sample)
    #         logli -= torch.sum(
    #             torch.log(self.scale * (1 - torch.tanh(sample) ** 2) + EPS),
    #             dim=-1)
    #     return sample, logli


        # scale = self.scale
        # self.scale = None  # Temporarily turn OFF.
        # sample = self.sample(dist_info)
        # self.scale = scale  # Turn it back ON, raw_sample into squash correction.
        # logli = self.log_likelihood(sample, dist_info)
        # if scale is not None:
        #     sample = scale * torch.tanh(sample)
        # return sample, logli

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
