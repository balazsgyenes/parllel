from gym import Env
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.wrappers import (
    TimeLimit,
)
from parllel.envs.gym import GymEnvWrapper

def make_env(
    max_episode_steps: int = 250,
) -> Env:
    env = CartPoleEnv()

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = GymEnvWrapper(env)

    return env