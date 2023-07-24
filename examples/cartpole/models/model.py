from __future__ import annotations

from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor

from parllel.torch.agents.categorical import DistParams, ModelOutputs
from parllel.torch.models import MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims


class CartPoleFfPgModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Discrete,
        hidden_sizes: int | Sequence[int] | None,
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

    def forward(self, observation: Tensor) -> ModelOutputs:
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)

        probs = F.softmax(self.pi(observation.view(T * B, -1)), dim=-1)
        value = self.value(observation.view(T * B, -1)).squeeze(-1)

        probs, value = restore_leading_dims((probs, value), lead_dim, T, B)

        return ModelOutputs(dist_params=DistParams(probs=probs), value=value)
