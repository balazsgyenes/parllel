from abc import ABC
import time

import parllel.logger as logger


class Runner(ABC):
    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.last_elapsed_steps = 0 # for fps calculation

    def log_progress(self, elapsed_steps: int) -> None:
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
