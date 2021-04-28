from typing import Dict, List, Tuple

from nptyping import NDArray

from .cage import Cage
from parllel.utils.traj_info import TrajInfo

class ParallelProcessCage(Cage):
    def __init__(self, EnvClass, env_kwargs) -> None:
        self.EnvClass = EnvClass
        self.env_kwargs = env_kwargs
        
        self._master_pipe = None
        self._follower_pipe = None
        self._step_buffer = None

    def initialize(self, step_buffer: NDArray) -> None:
        """Instantiate environment and subprocess, etc.
        """
        self._step_buffer = step_buffer

    def start_step(self) -> None:
        pass

    def await_step(self) -> NDArray:
        pass

    def shutdown(self) -> None:
        pass

    def collect_completed_trajs(self) -> List[TrajInfo]:
        pass
