from typing import Mapping, Sequence, TypedDict

from flax import linen as nn
import jax.numpy as jnp
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
        x = input.astype(jnp.float32)
        for size in self.actor_hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.relu(x)

        probs = nn.Dense(features=self.action_dim)
        # log_std = nn.Dense(features=self.action_dim)

        x = input
        for size in self.critic_hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.relu(x)
        value = nn.Dense(features=1)(x)

        return probs, value
