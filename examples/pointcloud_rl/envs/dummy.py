from __future__ import annotations

from typing import Any, SupportsFloat, SupportsInt

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .pointcloud import PointCloud


ObsType = np.ndarray
ActionType = SupportsInt


class DummyEnv(gym.Env):
    def __init__(
        self,
        prob_done: float,
    ) -> None:
        self.observation_space = PointCloud(
            max_num_points=50,
            low=-np.inf,
            high=np.inf,
            feature_shape=(3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.prob_done = prob_done

    def step(
        self, action: ActionType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs = self.observation_space.sample()
        done = self.np_random.random() < self.prob_done
        return (obs, 1.0, done, False, {})

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self._np_random = np.random.default_rng()

        return self.observation_space.sample(), {}


def build_dummy(
    prob_done: float,
) -> gym.Env:
    env = DummyEnv(prob_done=prob_done)

    return env
