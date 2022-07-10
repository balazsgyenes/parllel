from typing import Union

import torch

from parllel.torch.distributions.gaussian import Gaussian


class ClippedGaussian(Gaussian):
    def __init__(
            self,
            clip: Union[float, torch.Tensor, None],
            *args,
            **kwargs,
            ):
        super().__init__(*args, **kwargs)
        self.clip = clip

    def sample(self, dist_info):
        sample = super().sample(dist_info)
        if self.clip is not None:
            sample = torch.clamp(sample, -self.clip, self.clip)
        return sample

    def set_clip(self, clip: Union[float, torch.Tensor, None]):
        """Input value or ``None`` to turn OFF."""
        self.clip = clip  # Can be None.
