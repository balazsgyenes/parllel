from typing import Mapping, Sequence, TypedDict

import jax.numpy as jnp
from flax import linen as nn
from jax.typing import ArrayLike
from typing_extensions import NotRequired


class DistParams(TypedDict):
    probs: ArrayLike
    # log_std: ArrayLike


class ModelOutputs(TypedDict):
    dist_params: DistParams
    value: NotRequired[ArrayLike]


class ActorCriticModel(nn.Module):
    actor_hidden_sizes: Sequence[int]
    critic_hidden_sizes: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, input):
        input = input.astype(jnp.float32)
        x = input
        for size in self.actor_hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.relu(x)

        probs = nn.Dense(features=self.action_dim)(x).squeeze()
        # log_std = nn.Dense(features=self.action_dim)

        x = input
        for size in self.critic_hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.relu(x)
        value = nn.Dense(features=1)(x).squeeze()

        dist_params = DistParams(probs=probs)
        return ModelOutputs(dist_params=dist_params, value=value)
