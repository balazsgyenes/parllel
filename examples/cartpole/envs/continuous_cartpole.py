from gym import Env
from hera_gym.envs.cartpole import CartPoleEnv
from gym.wrappers import (
    TimeLimit,
)


def build_cartpole(
    max_episode_steps: int = 250,
) -> Env:
    env = CartPoleEnv(
        action_type="continuous",
        reward_type="sparse",
    )

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env
