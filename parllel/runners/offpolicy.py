from tqdm import tqdm

from parllel.algorithm import Algorithm
from parllel.samplers import Sampler, EvalSampler
from parllel.handlers import Agent
import parllel.logger as logger
from parllel.types import BatchSpec

from .runner import Runner


class OffPolicyRunner(Runner):
    def __init__(self,
        sampler: Sampler,
        agent: Agent,
        algorithm: Algorithm,
        batch_spec: BatchSpec,
        eval_sampler: EvalSampler,
        n_steps: int,
        log_interval_steps: int,
    ) -> None:
        super().__init__()

        self.sampler = sampler
        self.agent = agent
        self.algorithm = algorithm
        self.batch_spec = batch_spec
        self.eval_sampler = eval_sampler
        self.n_steps = n_steps

        self.n_iterations = int(n_steps // batch_spec.size)
        self.log_interval_iters = int(log_interval_steps // batch_spec.size)

    def run(self) -> None:
        logger.info("Starting training...")

        progress_bar = tqdm(total=self.n_steps, unit="steps")
        batch_size = self.batch_spec.size

        self.evaluate_agent(elapsed_steps=0, itr=0)
        
        for itr in range(self.n_iterations):
            elapsed_steps = itr * batch_size

            batch_samples, _ = self.sampler.collect_batch(elapsed_steps)

            algo_info = self.algorithm.optimize_agent(elapsed_steps, batch_samples)
            self.log_algo_info(algo_info)

            if (itr + 1) % self.log_interval_iters == 0:
                self.evaluate_agent(elapsed_steps=elapsed_steps, itr=itr)

            progress_bar.update(batch_size)

        progress_bar.close()        
        progress_bar = None
        logger.info("Finished training.")
        
    def evaluate_agent(self, elapsed_steps: int, itr: int) -> None:
        logger.debug("Evaluating agent.")
        eval_trajs = self.eval_sampler.collect_batch(elapsed_steps)
        self.log_completed_trajectories(eval_trajs)
        self.log_progress(elapsed_steps, itr)
