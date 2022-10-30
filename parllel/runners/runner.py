from abc import ABC
from ctypes import Union
from dataclasses import asdict
import time
from typing import Any, Dict, List

import numpy as np

from parllel.cages import TrajInfo
import parllel.logger as logger


class Runner(ABC):
    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.last_elapsed_steps = 0 # for fps calculation

    def log_progress(self, elapsed_steps: int, iteration: int) -> None:
        logger.record("time/iterations", iteration)
        
        time_elapsed = max(time.perf_counter() - self.start_time, 1e-6)
        fps = (elapsed_steps - self.last_elapsed_steps) / time_elapsed

        logger.record("time/fps", fps)
        logger.record("time/elapsed_time", time_elapsed)
        logger.record("time/elapsed_steps", elapsed_steps, do_not_write_to="tensorboard")

        logger.dump(step=elapsed_steps)

        # update elapsed steps for next fps calculation
        self.last_elapsed_steps = elapsed_steps

        # TODO: technically, the agent should be added to this base class
        logger.save_model(agent=self.agent)

    def log_completed_trajectories(self, trajectories: List[TrajInfo]) -> None:

        # ((key1, value1), (key2, value2), ...), ((key1, value1), (key2, value2), ...), ...
        trajectories = (asdict(traj).items() for traj in trajectories)

        # zip* -> ((key1, value1), (key1, value1), ...), ((key2, value2), (key2, value2), ...), ...
        for keys_and_values in zip(*trajectories):
            # (key1, value1), (key1, value1), ... -> (key1, key1, ...), (value1, value1, ...)
            keys, values = zip(*keys_and_values)
            key = keys[0]
            if key[0] == "_" or key == "discount":
                continue # do not log these "private" variables
            values = np.array(values)
            logger.record_mean("trajectory/" + key, values)

    def log_algo_info(self, info: Dict[str, Any]) -> None:
        for key, value in info.items():
            if isinstance(value, list):
                logger.record_mean("algo/" + key, value)
            else:
                logger.record("algo/" + key, value)
