from gymnasium import Env
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers.time_limit import TimeLimit


def build_cartpole(
    max_episode_steps: int,
) -> Env:
    env = CartPoleEnv()

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env
