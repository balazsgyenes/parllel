import time
from abc import ABC
from typing import Any, Sequence

import parllel.logger as logger
from parllel.agents import Agent
from parllel.cages import TrajInfo, zip_trajectories


class Runner(ABC):
    agent: Agent

    def __init__(self) -> None:
        # check that parllel.logger.init was called if WandB run exists
        logger.check_init()

        self.start_time = time.perf_counter()
        self.last_elapsed_steps = 0  # for fps calculation
        self.last_elapsed_time = 0.0

    def log_progress(self, elapsed_steps: int, iteration: int) -> None:
        logger.record("time/iterations", iteration)

        elapsed_time = max(time.perf_counter() - self.start_time, 1e-6)
        fps = (elapsed_steps - self.last_elapsed_steps) / (
            elapsed_time - self.last_elapsed_time
        )

        logger.record("time/fps", fps)
        logger.record("time/elapsed_time", elapsed_time)
        logger.record(
            "time/elapsed_steps", elapsed_steps, do_not_write_to="tensorboard"
        )

        logger.dump(step=elapsed_steps)

        # update elapsed steps and time for next fps calculation
        self.last_elapsed_steps = elapsed_steps
        self.last_elapsed_time = elapsed_time

        logger.save_model(agent=self.agent)

    def record_completed_trajectories(
        self,
        trajectories: Sequence[TrajInfo],
        prefix: str = "trajectory",
    ) -> None:
        for key, *values in zip_trajectories(*trajectories):
            if key[0] == "_":
                continue  # do not log these "private" variables
            logger.record_mean(prefix + "/" + key, values)

    def record_algo_info(self, info: dict[str, Any], prefix: str = "algo") -> None:
        for key, value in info.items():
            if isinstance(value, list):
                logger.record_mean(prefix + "/" + key, value)
            else:
                logger.record(prefix + "/" + key, value)
