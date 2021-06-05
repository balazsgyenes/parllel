from typing import Any, Tuple

import gym
from gym import spaces
import numpy as np
from nptyping import NDArray

from parllel.buffers import NamedTuple, NamedTupleClass

class DummyEnv(gym.Env):
    def __init__(self, episode_length: int) -> None:
        self._counter = None
        self._episode_length = episode_length
        self.observation_space = spaces.Box(low=0, high=episode_length, shape=())
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=())
        self._info_class = NamedTupleClass("info", ["action"])

    def step(self, action: NDArray) -> Tuple[NDArray, float, bool, NamedTuple]:
        self._counter += 1
        return (
            self._counter,
            self._counter,
            self._counter >= self._episode_length,
            self._info_class(action=action),
        )

    def reset(self) -> Any:
        self._counter = 0
        return self._counter
    