from typing import Dict, List, Tuple

from parllel.types.traj_info import TrajInfo
from parllel.buffers.buffer import Buffer
from .cage import Cage


class ParallelProcessCage(Cage):
    def __init__(self, EnvClass, env_kwargs, reset_after_batch: bool = False) -> None:
        self.EnvClass = EnvClass
        self.env_kwargs = env_kwargs
        self.reset_after_batch = reset_after_batch
        
        self._master_pipe = None
        self._follower_pipe = None
        self._step_buffer = None

    def initialize(self, step_buffer: Buffer) -> None:
        """Instantiate environment and subprocess, etc.
        """
        self._step_buffer = step_buffer
        self._completed_trajectories = []

    def step(self, action) -> None:
        pass

    def await_step(self) -> Buffer:
        pass

    def shutdown(self) -> None:
        pass

    def collect_completed_trajs(self) -> List[TrajInfo]:
        pass

    def random_step(self):
        """Take a step with a random action from the env's action space.
        """
        pass
