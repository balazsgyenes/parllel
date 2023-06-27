from gym import Env
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.wrappers import TimeLimit


def build_cartpole(
    max_episode_steps: int,
) -> Env:
    env = CartPoleEnv()

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env
