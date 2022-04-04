from itertools import count
from typing import Tuple

import gym
from gym import spaces
import numpy as np
from nptyping import NDArray

from parllel.buffers import NamedTuple, NamedTupleClass
from parllel.buffers.utils import buffer_method


class DummyEnv(gym.Env):
    def __init__(self, observation_space: gym.Space, episode_length: int, obs_start: int) -> None:
        self.observation_space = observation_space
        self._episode_length = episode_length
        self._obs_start = obs_start

        self._info_class = NamedTupleClass("info", ["action"])

    def step(self, action: NDArray) -> Tuple[NDArray, NDArray, bool, NamedTuple]:
        obs = self._get_obs()
        reward = next(self._traj_counter)
        done = reward >= self._episode_length
        return (
            obs,
            reward,
            done,
            self._info_class(action=action.copy()),
        )

    def reset(self) -> NDArray:
        self._obs_counter = count(self._obs_start)
        self._traj_counter = count()
        return self._get_obs()
    
    def _get_obs(self):
        obs = self.observation_space.sample()
        obs = buffer_method(obs, "fill", next(self._obs_counter))
        return obs
