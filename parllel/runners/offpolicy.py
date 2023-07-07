from tqdm import tqdm

from parllel.algorithm import Algorithm
from parllel.handlers import Agent
import parllel.logger as logger
from parllel.samplers import Sampler, EvalSampler
from parllel.types import BatchSpec

from .runner import Runner


class OffPolicyRunner(Runner):
    def __init__(self,
        sampler: Sampler,
        eval_sampler: EvalSampler,
        agent: Agent,
        algorithm: Algorithm,
        batch_spec: BatchSpec,
        n_steps: int,
        log_interval_steps: int,
        eval_interval_steps: int,
    ) -> None:
        super().__init__()

        self.sampler = sampler
        self.eval_sampler = eval_sampler
        self.agent = agent
        self.algorithm = algorithm
        self.batch_spec = batch_spec
        self.n_steps = n_steps

        self.n_iterations = max(1, int(n_steps // batch_spec.size))
        self.log_interval_iters = max(1, int(log_interval_steps // batch_spec.size))
        self.eval_interval_iters = max(1, int(eval_interval_steps // batch_spec.size))

    def run(self) -> None:
        logger.info("Starting training...")

        progress_bar = tqdm(total=self.n_steps, unit="steps")
        batch_size = self.batch_spec.size

        for itr in range(self.n_iterations):
            elapsed_steps = itr * batch_size

            if itr > 0 and itr % self.log_interval_iters == 0:
                self.log_progress(elapsed_steps, itr)

            # evaluates at 0th iteration
            if itr % self.eval_interval_iters == 0:
                self.evaluate_agent(elapsed_steps)

            batch_samples, completed_trajs = self.sampler.collect_batch(elapsed_steps)
            self.record_completed_trajectories(completed_trajs)

            algo_info = self.algorithm.optimize_agent(
                elapsed_steps,
                batch_samples,
            )
            self.record_algo_info(algo_info)

            progress_bar.update(batch_size)

        # log final progress
        elapsed_steps = self.n_iterations * batch_size
        self.evaluate_agent(elapsed_steps)

        progress_bar.close()
        logger.info("Finished training.")
        if logger.log_dir is not None:
            logger.info(f"Log files saved to {logger.log_dir}")

    def evaluate_agent(self, elapsed_steps: int) -> None:
        logger.debug("Evaluating agent.")
        eval_trajs = self.eval_sampler.collect_batch(elapsed_steps)
        self.record_eval_trajectories(eval_trajs)
