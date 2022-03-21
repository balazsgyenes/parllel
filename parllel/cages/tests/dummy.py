import time
from typing import Any, Dict, List, Optional, Tuple

import gym
from nptyping import NDArray


class DummyEnv(gym.Env):
    def __init__(self,
                 step_duration: float,
                 observation_space: gym.Space,
                 action_space: gym.Space = None,
                 ) -> None:
        self._step_duration = step_duration
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, action: NDArray) -> Tuple[NDArray, float, bool, Dict[str, Any]]:
        time.sleep(self._step_duration)
        return self.observation_space.sample(), 1., False, {}

    def reset(self) -> NDArray:
        return self.observation_space.sample()

    def seed(self, seed: Optional[int]) -> List[int]:
        self.observation_space.seed(seed)
        if self.action_space is not None:
            self.action_space.seed(seed+1)
        return [seed]
