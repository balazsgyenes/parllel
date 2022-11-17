from tqdm import tqdm

from parllel.algorithm import Algorithm
from parllel.handlers.agent import Agent
import parllel.logger as logger
from parllel.samplers.sampler import Sampler
from parllel.types import BatchSpec

from .runner import Runner


class OnPolicyRunner(Runner):
    def __init__(self,
        sampler: Sampler,
        agent: Agent,
        algorithm: Algorithm,
        n_steps: int,
        batch_spec: BatchSpec,
        log_interval_steps: int,
    ) -> None:
        super().__init__()

        self.sampler = sampler
        self.agent = agent
        self.algorithm = algorithm
        self.n_steps = n_steps
        self.batch_spec = batch_spec

        self.n_iterations = max(1, int(n_steps // batch_spec.size))
        self.log_interval_iters = max(1, int(log_interval_steps // batch_spec.size))

    def run(self) -> None:
        logger.info("Starting training...")
        
        progress_bar = tqdm(total=self.n_steps, unit="steps")
        batch_size = self.batch_spec.size

        for itr in range(self.n_iterations):
            elapsed_steps = itr * batch_size

            batch_samples, completed_trajs = self.sampler.collect_batch(
                elapsed_steps,
            )
            self.record_completed_trajectories(completed_trajs)

            algo_info = self.algorithm.optimize_agent(
                elapsed_steps,
                batch_samples,
            )
            self.record_algo_info(algo_info)

            if (itr + 1) % self.log_interval_iters == 0:
                self.log_progress(elapsed_steps, itr)

            progress_bar.update(batch_size)

        progress_bar.close()
        progress_bar = None
        logger.info("Finished training.")
