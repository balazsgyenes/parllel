from abc import ABC
import time
from typing import Any, Dict, List

import numpy as np

from parllel.cages import TrajInfo, zip_trajectories
import parllel.logger as logger


class Runner(ABC):
    def __init__(self) -> None:
        # check that parllel.logger.init was called if WandB run exists
        logger.check_init()

        self.start_time = time.perf_counter()
        self.last_elapsed_steps = 0 # for fps calculation
        self.last_elapsed_time = 0.

    def log_progress(self, elapsed_steps: int, iteration: int) -> None:
        logger.record("time/iterations", iteration)
        
        elapsed_time = max(time.perf_counter() - self.start_time, 1e-6)
        fps = ((elapsed_steps - self.last_elapsed_steps)
            / (elapsed_time - self.last_elapsed_time))

        logger.record("time/fps", fps)
        logger.record("time/elapsed_time", elapsed_time)
        logger.record("time/elapsed_steps", elapsed_steps, do_not_write_to="tensorboard")

        logger.dump(step=elapsed_steps)

        # update elapsed steps and time for next fps calculation
        self.last_elapsed_steps = elapsed_steps
        self.last_elapsed_time = elapsed_time

        # TODO: technically, the agent should be added to this base class
        logger.save_model(agent=self.agent)

    def record_completed_trajectories(self, trajectories: List[TrajInfo]) -> None:
        for key, *values in zip_trajectories(*trajectories):
            if key[0] == "_":
                continue # do not log these "private" variables
            values = np.array(values)
            logger.record_mean("trajectory/" + key, values)

    def record_algo_info(self, info: Dict[str, Any]) -> None:
        for key, value in info.items():
            if isinstance(value, list):
                logger.record_mean("algo/" + key, value)
            else:
                logger.record("algo/" + key, value)
