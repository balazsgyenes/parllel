from typing import Tuple

from gym import Env, spaces
import numpy as np

from pointcloud import PointCloud


class DummyEnv(Env):
    def __init__(self,
        prob_done: float,
    ) -> None:
        self.observation_space = PointCloud(
            max_num_points=50,
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.prob_done = prob_done

        self.seed()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs = self.observation_space.sample()
        done = self.rng.random() < self.prob_done
        return (obs, 1.0, done, {})

    def reset(self) -> np.ndarray:
        return self.observation_space.sample()


def build_dummy(
    prob_done: float,
) -> Env:
    env = DummyEnv(prob_done=prob_done)

    return env
