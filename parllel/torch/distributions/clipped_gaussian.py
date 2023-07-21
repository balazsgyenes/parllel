from __future__ import annotations

from typing import SupportsFloat

import torch

from parllel.torch.distributions.gaussian import Gaussian, DistParams


class ClippedGaussian(Gaussian):
    def __init__(
            self,
            clip: SupportsFloat | None,
            *args,
            **kwargs,
            ):
        super().__init__(*args, **kwargs)
        self.clip = clip

    def sample(self, dist_params: DistParams):
        sample = super().sample(dist_params)
        if self.clip is not None:
            sample = torch.clamp(sample, -self.clip, self.clip)
        return sample

    def set_clip(self, clip: SupportsFloat | None):
        """Input value or ``None`` to turn OFF."""
        self.clip = clip  # Can be None.
