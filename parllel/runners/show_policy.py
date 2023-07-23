import parllel.logger as logger
from parllel.agents import Agent
from parllel.samplers import EvalSampler

from .runner import Runner


class ShowPolicy(Runner):
    def __init__(self,
        sampler: EvalSampler,
        agent: Agent,
    ) -> None:
        super().__init__()

        self.sampler = sampler
        self.agent = agent

    def run(self) -> None:
        logger.info("Showing policy...")

        eval_trajs = self.sampler.collect_batch(elapsed_steps=0)

        self.record_completed_trajectories(eval_trajs)
        self.log_progress(elapsed_steps=0, iteration=0)

        logger.info("Finished.")
