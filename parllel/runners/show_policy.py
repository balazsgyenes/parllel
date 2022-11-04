from parllel.handlers.agent import Agent
import parllel.logger as logger
from parllel.samplers import EvalSampler

from .runner import Runner


class ShowPolicy(Runner):
    def __init__(self,
        sampler: EvalSampler,
        agent: Agent,
    ) -> None:

        self.sampler = sampler
        self.agent = agent

    def run(self) -> None:
        logger.info("Showing policy...")

        eval_trajs = self.sampler.collect_batch(elapsed_steps=0)

        self.log_completed_trajectories(eval_trajs)
        self.log_progress(elapsed_steps=0, itr=0)

        logger.info("Finished.")
