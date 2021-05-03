from typing import Dict, List, Callable
import multiprocessing as mp

from parllel.envs.collections import EnvStep
from parllel.types.traj_info import TrajInfo
from .cage import Cage


class ParallelProcessCage(Cage):
    def initialize(self) -> None:
        """Instantiate environment and subprocess, etc.
        """
        super().initialize()

    def step(self, action) -> None:
        raise NotImplementedError

    def await_step(self) -> EnvStep:
        raise NotImplementedError

    def random_step(self):
        """Take a step with a random action from the env's action space.
        """
        raise NotImplementedError
    
    def collect_completed_trajs(self) -> List[TrajInfo]:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
