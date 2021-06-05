from typing import Any, Dict, Tuple

import gym
from gym import spaces
import numpy as np
from nptyping import NDArray

class DummyEnv(gym.Env):
    def __init__(self, episode_length: int) -> None:
        self._counter = None
        self._episode_length = episode_length
        self.observation_space = spaces.Box(low=0, high=episode_length)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

    def step(self, action: NDArray) -> Tuple[NDArray, float, bool, Dict[str, Any]]:
        self._counter += 1
        return (
            self._counter,
            self._counter,
            self._counter >= self._episode_length,
            {"action": action},
        )

    def reset(self) -> Any:
        self._counter = 0
        return self._counter
    