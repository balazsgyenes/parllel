from typing import Literal

from gymnasium import Env
from gymnasium.wrappers.time_limit import TimeLimit

from .cartpole import ExtendedCartPoleEnv


def build_cartpole(
    max_episode_steps: int = 250,
    reward_type: Literal["dense", "sparse"] = "dense",
) -> Env:
    env = ExtendedCartPoleEnv(
        action_type="continuous",
        reward_type=reward_type,
    )

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env
