import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, Env


class HashableActionWrapper(ActionWrapper):
    def action(self, action: np.ndarray) -> int:
        return action.item()


def build_frozen_lake(**kwargs) -> Env:
    env = gym.make("FrozenLake-v1", **kwargs)
    # env needs hashable actions: turn ndarray to int
    env = HashableActionWrapper(env)
    return env
