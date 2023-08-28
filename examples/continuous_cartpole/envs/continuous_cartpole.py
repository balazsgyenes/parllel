from typing import Literal

from gymnasium import Env
from gymnasium.wrappers import TimeLimit, TransformReward  # type: ignore

from .cartpole import ExtendedCartPoleEnv


def build_cartpole(
    max_episode_steps: int = 250,
    reward_type: Literal["dense", "sparse"] = "dense",
    reward_scale: float = 1.0,
) -> Env:
    env = ExtendedCartPoleEnv(
        action_type="continuous",
        reward_type=reward_type,
    )

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = TransformReward(
        env,
        lambda r: r * reward_scale,
    )

    return env
