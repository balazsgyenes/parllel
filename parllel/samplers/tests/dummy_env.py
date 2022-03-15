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
        self.observation_space = spaces.Box(low=0, high=episode_length, shape=(), dtype=np.int32)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.int32)
        self._info_class = NamedTupleClass("info", ["action"])

    def step(self, action: NDArray) -> Tuple[NDArray, NDArray, NDArray, NamedTuple]:
        self._counter += 1
        return (
            np.array(self._counter, dtype=np.int32),
            np.array(self._counter, dtype=np.int32),
            np.array(self._counter >= self._episode_length, dtype=np.bool_),
            self._info_class(action=action.copy()),
        )

    def reset(self) -> NDArray:
        self._counter = 0
        return np.array(self._counter, dtype=np.int32)
    