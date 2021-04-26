from typing import Dict, List, Tuple

from nptyping import NDArray

from parllel.utils.traj_info import TrajInfo

class ParallelProcessCage:
    def __init__(self, EnvClass, env_kwargs) -> None:
        self.EnvClass = EnvClass
        self.env_kwargs = env_kwargs
        
        self._barrier = None
        self._batch_buffer = None

    def initialize(self) -> None:
        """Instantiate environment and subprocess, etc.
        """
        pass
    
    def get_example_output(self) -> Tuple[NDArray, NDArray, bool, Dict]:
        """Get an example of the output from the env for building buffers.
        """
        pass

    def start_step(self) -> None:
        pass

    def await_step(self) -> None:
        pass

    def start_reset(self) -> None:
        pass

    def await_reset(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def collect_completed_trajs(self) -> List[TrajInfo]:
        pass

    @property
    def batch_buffer(self):
        return self._batch_buffer