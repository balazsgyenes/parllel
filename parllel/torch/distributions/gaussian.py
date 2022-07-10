import numpy as np
import torch

from parllel.buffers import NamedArrayTupleClass

from .distribution import Distribution


EPS = 1e-8

DistInfo = NamedArrayTupleClass("DistInfo", ["mean"])
DistInfoStd = NamedArrayTupleClass("DistInfoStd", ["mean", "log_std"])


class Gaussian(Distribution):
    """Multivariate Gaussian with independent variables (diagonal covariance).
    Standard deviation can be provided, as scalar or value per dimension, or it
    will be drawn from the dist_info (possibly learnable), where it is expected
    to have a value per each dimension.
    Noise clipping or sample clipping optional during sampling, but not
    accounted for in formulas (e.g. entropy).
    Clipping of standard deviation optional and accounted in formulas.
    Squashing of samples to squash * tanh(sample) is optional and accounted for
    in log_likelihood formula but not entropy.
    """

    def __init__(
            self,
            dim,
            std=None,
            noise_clip=None,
            min_std=None,
            max_std=None,
            ):
        """Saves input arguments."""
        self._dim = dim
        self.set_std(std)
        self.noise_clip = noise_clip
        self.min_std = min_std
        self.max_std = max_std
        self.min_log_std = np.log(min_std) if min_std is not None else None
        self.max_log_std = np.log(max_std) if max_std is not None else None

    @property
    def dim(self):
        return self._dim

    def kl(self, old_dist_info, new_dist_info):
        old_mean = old_dist_info.mean
        new_mean = new_dist_info.mean
        # Formula: {[(m1 - m2)^2 + (s1^2 - s2^2)] / (2*s2^2)} + ln(s1/s2)
        num = (old_mean - new_mean) ** 2
        if self.std is None:
            old_log_std = old_dist_info.log_std
            new_log_std = new_dist_info.log_std
            if self.min_std is not None or self.max_std is not None:
                old_log_std = torch.clamp(old_log_std, min=self.min_log_std,
                    max=self.max_log_std)
                new_log_std = torch.clamp(new_log_std, min=self.min_log_std,
                    max=self.max_log_std)
            old_std = torch.exp(old_log_std)
            new_std = torch.exp(new_log_std)
            num += old_std ** 2 - new_std ** 2
            den = 2 * new_std ** 2 + EPS
            vals = num / den + new_log_std - old_log_std
        else:
            den = 2 * self.std ** 2 + EPS
            vals = num / den
        return torch.sum(vals, dim=-1)

    def entropy(self, dist_info):
        """Uses ``self.std`` unless that is None, then will get log_std from dist_info.
        """
        if self.std is None:
            log_std = dist_info.log_std
            if self.min_log_std is not None or self.max_log_std is not None:
                log_std = torch.clamp(log_std, min=self.min_log_std,
                    max=self.max_log_std)
        else:
            # shape = dist_info.mean.shape[:-1]
            # log_std = torch.log(self.std).repeat(*shape, 1)
            log_std = torch.log(self.std)  # Shape broadcast in following formula.
        return torch.sum(log_std + np.log(np.sqrt(2 * np.pi * np.e)),
            dim=-1)

    def log_likelihood(self, x, /, dist_info):
        """
        Uses ``self.std`` unless that is None, then uses log_std from dist_info.
        """
        mean = dist_info.mean
        if self.std is None:
            log_std = dist_info.log_std
            if self.min_log_std is not None or self.max_log_std is not None:
                log_std = torch.clamp(log_std, min=self.min_log_std,
                    max=self.max_log_std)
            std = torch.exp(log_std)
        else:
            std, log_std = self.std, torch.log(self.std)
        z = (x - mean) / (std + EPS)
        logli = -(torch.sum(log_std + 0.5 * z ** 2, dim=-1) + 0.5 * self.dim * np.log(2 * np.pi))
        return logli

    def likelihood_ratio(self, x, /, old_dist_info, new_dist_info):
        # TODO: this is stupid. fix it
        logli_old = self.log_likelihood(x, old_dist_info)
        logli_new = self.log_likelihood(x, new_dist_info)
        return torch.exp(logli_new - logli_old)

    def sample(self, dist_info):
        """
        Generate random samples using ``torch.normal``, from
        ``dist_info.mean``. Uses ``self.std`` unless it is ``None``, then uses
        ``dist_info.log_std``.
        """
        mean = dist_info.mean
        if self.std is None:
            log_std = dist_info.log_std
            if self.min_log_std is not None or self.max_log_std is not None:
                log_std = torch.clamp(log_std, min=self.min_log_std,
                    max=self.max_log_std)
            std = torch.exp(log_std)
        else:
            # shape = mean.shape[:-1]
            # std = self.std.repeat(*shape, 1).to(mean.device)
            std = self.std.to(mean.device)
        # For reparameterization trick: mean + std * N(0, 1)
        # (Also this gets noise on same device as mean.)
        noise = std * torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
        # noise = torch.normal(mean=0, std=std)
        if self.noise_clip is not None:
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        sample = mean + noise
        # Other way to do reparameterization trick:
        # dist = torch.distributions.Normal(mean, std)
        # sample = dist.rsample()
        return sample

    def set_noise_clip(self, noise_clip):
        """Input value or ``None`` to turn OFF."""
        self.noise_clip = noise_clip  # Can be None.

    def set_std(self, std):
        """
        Input value, which can be same shape as action space, or else broadcastable
        up to that shape, or ``None`` to turn OFF and use ``dist_info.log_std`` in
        other methods.
        """
        if std is not None:
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std).float()  # Can be size == 1 or dim.
            # Used to have, but shape of std should broadcast everywhere needed:
            # if std.numel() == 1:
            #     std = std * torch.ones(self.dim).float()  # Make it size dim.
            assert std.numel() in (self.dim, 1)
        self.std = std
