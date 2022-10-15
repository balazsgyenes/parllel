from abc import ABC

import parllel.logger as logger


class Runner(ABC):
    def log_progress(self, elapsed_steps: int) -> None:
        logger.dump(step=elapsed_steps)

        if logger.model_save_path is not None:
            # TODO: technically, the agent should be added to this base class
            self.agent.save_model(path=logger.model_save_path)
