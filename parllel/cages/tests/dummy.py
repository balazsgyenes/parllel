from __future__ import annotations

import time

import gymnasium as gym
import numpy as np


class DummyEnv(gym.Env):
    def __init__(
        self,
        step_duration: float,
        observation_space: gym.Space,
        action_space: gym.Space | None = None,
    ) -> None:
        self._step_duration = step_duration
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        time.sleep(self._step_duration)
        return self.observation_space.sample(), 1.0, False, False, {}

    def reset(self) -> np.ndarray:
        return self.observation_space.sample()

    def seed(self, seed: int | None) -> list[int]:
        seeds = self.observation_space.seed(seed)
        more_seeds = self.action_space.seed(seeds[0] + 1)
        return seeds + more_seeds
