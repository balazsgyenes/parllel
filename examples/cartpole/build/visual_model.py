from typing import List, Union

from gym import spaces
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from parllel.torch.agents.categorical import ModelOutputs
from parllel.torch.models import Conv2dHeadModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims


class VisualCartPoleFfCategoricalPgModel(nn.Module):
    def __init__(self,
                 obs_space: spaces.Box,
                 action_space: spaces.Discrete,
                 channels: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 paddings: List[int],
                 use_maxpool: bool,
                 hidden_sizes: Union[int, List[int], None],
                 nonlinearity: nn.Module,
                 ) -> None:
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        assert obs_space.dtype == np.uint8
        image_shape = tuple(obs_space.shape)

        assert isinstance(action_space, spaces.Discrete)
        n_actions = action_space.n

        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_maxpool=use_maxpool,
            hidden_sizes=hidden_sizes,
            nonlinearity=nonlinearity
        )
        self.pi = nn.Linear(self.conv.output_size, n_actions)
        self.value = nn.Linear(self.conv.output_size, 1)

    def forward(self, observation):

        image = observation.type(torch.float)  # Requires torch.uint8 inputs
        image = image.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)

        encoding = self.conv(image.view(T * B, *img_shape))
        pi = F.softmax(self.pi(encoding), dim=-1)
        value = self.value(encoding).squeeze(-1)

        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)

        return ModelOutputs(pi = pi, value = value)