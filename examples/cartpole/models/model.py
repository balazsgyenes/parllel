from typing import Sequence, Union

import torch.nn.functional as F
from gymnasium import spaces
from torch import nn

from parllel.torch.agents.categorical import ModelOutputs
from parllel.torch.models import MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims


class CartPoleFfPgModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Discrete,
        hidden_sizes: Union[int, Sequence[int], None],
        hidden_nonlinearity: str,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        obs_shape = obs_space.shape[0]

        assert isinstance(action_space, spaces.Discrete)
        n_actions = int(action_space.n)

        hidden_nonlinearity = getattr(nn, hidden_nonlinearity)

        self.pi = MlpModel(
            input_size=obs_shape,
            hidden_sizes=hidden_sizes,
            output_size=n_actions,
            hidden_nonlinearity=hidden_nonlinearity,
        )

        self.value = MlpModel(
            input_size=obs_shape,
            hidden_sizes=hidden_sizes,
            output_size=1,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(self, observation):
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)

        prob = F.softmax(self.pi(observation.view(T * B, -1)), dim=-1)
        value = self.value(observation.view(T * B, -1)).squeeze(-1)

        prob, value = restore_leading_dims((prob, value), lead_dim, T, B)

        return ModelOutputs(prob=prob, value=value)
