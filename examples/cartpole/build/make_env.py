from gym import Env
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.wrappers import (
    TimeLimit,
)


def make_env(
    max_episode_steps: int = 250,
) -> Env:
    env = CartPoleEnv()

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env