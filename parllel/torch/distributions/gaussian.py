from __future__ import annotations

from typing import SupportsFloat, TypedDict

import numpy as np
import torch
from torch import Tensor

from .distribution import Distribution

MIN_LOG_STD = -20.0
MAX_LOG_STD = 2.0
EPS = 1e-8


class DistParams(TypedDict):
    mean: Tensor
    log_std: Tensor


class Gaussian(Distribution):
    """Multivariate Gaussian with independent variables (diagonal covariance).
    Standard deviation can be provided, as scalar or value per dimension, or it
    will be drawn from the dist_params (possibly learnable), where it is expected
    to have a value per each dimension.
    Noise clipping or sample clipping optional during sampling, but not
    accounted for in formulas (e.g. entropy).
    Clipping of standard deviation optional and accounted in formulas.
    Squashing of samples to squash * tanh(sample) is optional and accounted for
    in log_likelihood formula but not entropy.
    """

    def __init__(
        self,
        dim: int,
        fixed_std: SupportsFloat | None = None,
        noise_clip: SupportsFloat | None = None,
        min_log_std: SupportsFloat | None = MIN_LOG_STD,
        max_log_std: SupportsFloat | None = MAX_LOG_STD,
        deterministic_eval_mode: bool = False,
    ):
        """Saves input arguments."""
        self._dim = dim
        self.device = None
        self.set_fixed_std(fixed_std)
        self.set_noise_clip(noise_clip)
        self.set_min_log_std(min_log_std)
        self.set_max_log_std(max_log_std)
        self.deterministic_eval_mode = deterministic_eval_mode

    def to_device(self, device: torch.device) -> None:
        self.device = device
        self.set_fixed_std(self.fixed_std)
        self.set_noise_clip(self.noise_clip)
        self.set_min_log_std(self.min_log_std)
        self.set_max_log_std(self.max_log_std)

    @property
    def dim(self) -> int:
        return self._dim

    def sample(self, dist_params: DistParams) -> Tensor:
        """Generate differentiable random samples. Uses `self.fixed_std`
        unless it is `None`, then uses `dist_params["log_std"]`.
        """
        mean = dist_params["mean"]
        if self.deterministic_eval_mode and self.mode == "eval":
            noise = torch.zeros_like(mean)
        else:
            if self.fixed_std is None:
                log_std = dist_params["log_std"]
                if self.min_log_std is not None or self.max_log_std is not None:
                    log_std = torch.clamp(
                        log_std, min=self.min_log_std, max=self.max_log_std
                    )
                std = torch.exp(log_std)
            else:
                std = self.fixed_std
            # For reparameterization trick: mean + std * N(0, 1)
            # (Also this gets noise on same device as mean.)
            noise = std * torch.normal(0, torch.ones_like(mean))
            # noise = torch.normal(mean=0, std=std)
            if self.noise_clip is not None:
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

        sample = mean + noise
        # Other way to do reparameterization trick:
        # dist = torch.distributions.Normal(mean, std)
        # sample = dist.rsample()
        return sample

    def kl(self, old_dist_params: DistParams, new_dist_params: DistParams) -> Tensor:
        old_mean = old_dist_params["mean"]
        new_mean = new_dist_params["mean"]
        # Formula: {[(m1 - m2)^2 + (s1^2 - s2^2)] / (2*s2^2)} + ln(s1/s2)
        num = (old_mean - new_mean) ** 2
        if self.fixed_std is None:
            old_log_std = old_dist_params["log_std"]
            new_log_std = new_dist_params["log_std"]
            if self.min_log_std is not None or self.max_log_std is not None:
                old_log_std = torch.clamp(
                    old_log_std, min=self.min_log_std, max=self.max_log_std
                )
                new_log_std = torch.clamp(
                    new_log_std, min=self.min_log_std, max=self.max_log_std
                )
            old_std = torch.exp(old_log_std)
            new_std = torch.exp(new_log_std)
            num += old_std**2 - new_std**2
            den = 2 * new_std**2 + EPS
            vals = num / den + new_log_std - old_log_std
        else:
            den = 2 * self.fixed_std**2 + EPS
            vals = num / den
        return torch.sum(vals, dim=-1)

    def log_likelihood(self, x: Tensor, /, dist_params: DistParams) -> Tensor:
        """Uses `self.fixed_std` unless it is `None`, then uses
        `dist_params["log_std"]`.
        """
        mean = dist_params["mean"]
        if self.fixed_std is None:
            log_std = dist_params["log_std"]
            if self.min_log_std is not None or self.max_log_std is not None:
                log_std = torch.clamp(
                    log_std, min=self.min_log_std, max=self.max_log_std
                )
            std = torch.exp(log_std)
        else:
            std, log_std = self.fixed_std, torch.log(self.fixed_std)
        z = (x - mean) / (std + EPS)
        logli = -(
            torch.sum(log_std + 0.5 * z**2, dim=-1)
            + 0.5 * self.dim * np.log(2 * np.pi)
        )
        return logli

    def likelihood_ratio(
        self,
        x: Tensor,
        /,
        old_dist_params: DistParams,
        new_dist_params: DistParams,
    ) -> Tensor:
        if self.fixed_std is None:
            # L_n/L_o = s_o/s_n * exp(-1/2 * (z_n^2 - z_o^2))
            # where z = (x - mu) / s
            old_log_std = old_dist_params["log_std"]
            new_log_std = new_dist_params["log_std"]
            if self.min_log_std is not None or self.max_log_std is not None:
                old_log_std = torch.clamp(
                    old_log_std, min=self.min_log_std, max=self.max_log_std
                )
                new_log_std = torch.clamp(
                    new_log_std, min=self.min_log_std, max=self.max_log_std
                )
            old_std = torch.exp(old_log_std)
            new_std = torch.exp(new_log_std)

            old_z = (x - old_dist_params["mean"]) / (old_std + EPS)
            new_z = (x - new_dist_params["mean"]) / (new_std + EPS)

            ratios = old_std / new_std * torch.exp(-(new_z**2 - old_z**2) / 2)

        else:
            # L_n/L_o = exp(-1/2 * (X_n^2 - X_o^2) / s^2)
            # where X = x - mu
            old_X = x - old_dist_params["mean"]
            new_X = x - new_dist_params["mean"]

            ratios = torch.exp(-(new_X**2 - old_X**2) / 2 / self.fixed_std**2)

        return torch.sum(ratios, dim=-1)

    def entropy(self, dist_params: DistParams) -> Tensor:
        """Uses `self.fixed_std` unless it is `None`, then uses
        `dist_params["log_std"]`.
        """
        if self.fixed_std is None:
            log_std = dist_params["log_std"]
            if self.min_log_std is not None or self.max_log_std is not None:
                log_std = torch.clamp(
                    log_std, min=self.min_log_std, max=self.max_log_std
                )
        else:
            # shape = dist_params["mean"].shape[:-1]
            # log_std = torch.log(self.fixed_std).repeat(*shape, 1)
            log_std = torch.log(self.fixed_std)  # Shape broadcast in following formula.
        return torch.sum(log_std + np.log(np.sqrt(2 * np.pi * np.e)), dim=-1)

    def set_noise_clip(self, noise_clip: SupportsFloat | None) -> None:
        """Input value or ``None`` to turn OFF."""
        noise_clip = clean_optionalfloatlike(noise_clip, self.dim, self.device)
        self.noise_clip = noise_clip

    def set_fixed_std(self, std: SupportsFloat | None) -> None:
        """
        Input value, which can be same shape as action space, or else broadcastable
        up to that shape, or ``None`` to turn OFF and use ``dist_params["log_std"]`` in
        other methods.
        """
        std = clean_optionalfloatlike(std, self.dim, self.device)
        self.fixed_std = std

    def set_min_log_std(self, min_log_std: SupportsFloat | None) -> None:
        min_log_std = clean_optionalfloatlike(min_log_std, self.dim, self.device)
        self.min_log_std = min_log_std

    def set_max_log_std(self, max_log_std: SupportsFloat | None) -> None:
        max_log_std = clean_optionalfloatlike(max_log_std, self.dim, self.device)
        self.max_log_std = max_log_std


def clean_optionalfloatlike(
    value: SupportsFloat | None,
    dim: int,
    device: torch.device | None,
) -> torch.Tensor | None:
    if value is not None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        assert value.shape in ((dim,), (1,), ())
        value = value.float()
        if device is not None:
            value = value.to(device)
    return value
