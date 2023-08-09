from typing import Mapping, Sequence, TypedDict

from flax import linen as nn


class ActorCriticModel(nn.Module):
    actor_hiddden_sizes: Sequence[int]
    critic_hidden_sizes: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, input):
        x = input
        for size in self.actor_hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.relu(x)

        mean = nn.Dense(features=self.action_dim)
        log_std = nn.Dense(features=self.action_dim)

        x = input
        for size in self.critic_hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.relu(x)
        value = nn.Dense(features=1)(x)

        return mean, log_std, value
