from __future__ import annotations

from typing import Sequence

from tqdm import tqdm

import parllel.logger as logger
from parllel.agents import Agent
from parllel.algorithm import Algorithm
from parllel.callbacks import Callback
from parllel.samplers import EvalSampler, Sampler
from parllel.types import BatchSpec

from .runner import Runner


class RLRunner(Runner):
    def __init__(
        self,
        sampler: Sampler,
        agent: Agent,
        algorithm: Algorithm,
        batch_spec: BatchSpec,
        n_steps: int,
        log_interval_steps: int,
        eval_sampler: EvalSampler | None = None,
        eval_interval_steps: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        logger_rollout_prefix: str = "rollout",
        logger_eval_prefix: str = "eval",
        logger_algo_prefix: str = "algo",
    ) -> None:
        super().__init__()

        self.sampler = sampler
        self.agent = agent
        self.algorithm = algorithm
        self.batch_spec = batch_spec
        self.n_steps = int(n_steps)
        self.eval_sampler = eval_sampler
        self.callbacks = list(callbacks) if callbacks is not None else []
        self.logger_rollout_prefix = logger_rollout_prefix
        self.logger_eval_prefix = logger_eval_prefix
        self.logger_algo_prefix = logger_algo_prefix

        self.n_iterations = max(1, int(n_steps // batch_spec.size))
        self.log_interval_iters = max(1, int(log_interval_steps // batch_spec.size))
        if eval_sampler is not None:
            if eval_interval_steps is not None:
                self.eval_interval_iters = max(
                    1, int(eval_interval_steps // batch_spec.size)
                )
            else:
                raise ValueError(
                    "Please specify `eval_interval_steps`, which tells the runner to evaluate the policy every N training steps."
                )

    def run(self) -> None:
        logger.info(f"{type(self).__name__}: Starting training...")
        if logger.log_dir is not None:
            logger.info(f"{type(self).__name__}: Saving log files to {logger.log_dir}")

        progress_bar = tqdm(total=self.n_steps, unit="steps")
        batch_size = self.batch_spec.size

        for itr in range(self.n_iterations):
            elapsed_steps = itr * batch_size

            if self.eval_sampler is not None and itr % self.eval_interval_iters == 0:
                self.evaluate_agent(elapsed_steps)

            # logs at 0th iteration only if there is an eval sampler
            if itr % self.log_interval_iters == 0 and (
                itr > 0 or self.eval_sampler is not None
            ):
                self.log_progress(elapsed_steps, itr)

            logger.debug(f"{type(self).__name__}: Collecting batch #{itr + 1}...")
            for callback in self.callbacks:
                callback.pre_sampling(elapsed_steps)
            batch_samples, completed_trajs = self.sampler.collect_batch(elapsed_steps)
            for callback in self.callbacks:
                callback.post_sampling(elapsed_steps)
            self.record_completed_trajectories(
                completed_trajs,
                prefix=self.logger_rollout_prefix,
            )
            logger.debug(
                f"{type(self).__name__}: Finished collecting batch #{itr + 1}."
            )

            logger.debug(f"{type(self).__name__}: Optimizing agent...")
            for callback in self.callbacks:
                callback.pre_optimization(elapsed_steps)
            algo_info = self.algorithm.optimize_agent(
                elapsed_steps,
                batch_samples,
            )
            for callback in self.callbacks:
                callback.post_optimization(elapsed_steps)
            self.record_algo_info(algo_info, prefix=self.logger_algo_prefix)
            logger.debug(f"{type(self).__name__}: Finished optimizing agent.")

            progress_bar.update(batch_size)

        # log final progress
        elapsed_steps = self.n_iterations * batch_size
        if self.eval_sampler is not None:
            self.evaluate_agent(elapsed_steps)
        self.log_progress(self.n_iterations * batch_size, self.n_iterations)

        progress_bar.close()
        # TODO: replace with logger.finish method
        logger.info(f"{type(self).__name__}: Finished training.")
        if logger.log_dir is not None:
            logger.info(f"{type(self).__name__}: Log files saved to {logger.log_dir}")

    def evaluate_agent(self, elapsed_steps: int) -> None:
        assert self.eval_sampler is not None
        logger.info(f"{type(self).__name__}: Evaluating agent...")
        for callback in self.callbacks:
            callback.pre_evaluation(elapsed_steps)
        eval_trajs = self.eval_sampler.collect_batch(elapsed_steps)
        for callback in self.callbacks:
            callback.post_evaluation(elapsed_steps)
        self.record_completed_trajectories(eval_trajs, prefix=self.logger_eval_prefix)
        logger.info(f"{type(self).__name__}: Finished evaluating agent.")
