from __future__ import annotations

from tqdm import tqdm

# import parllel.logger as logger
from parllel.jax import agent
from parllel.runners.runner import Runner
from parllel.samplers import EvalSampler, Sampler
from parllel.types import BatchSpec
import jax


class JaxRunner(Runner):
    def __init__(
        self,
        sampler,
        algorithm,
        batch_spec,
        n_steps,
        log_interval_steps,
        eval_sampler = None,
        eval_interval_steps = None,
    ):
        super().__init__()
        self.sampler = sampler
        self.eval_sampler = eval_sampler
        self.algorithm = algorithm
        self.batch_spec = batch_spec
        self.n_steps = n_steps

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

    def run(self, state) -> None:
        # logger.info(f"{type(self).__name__}: Starting training...")

        progress_bar = tqdm(total=self.n_steps, unit="steps")
        batch_size = self.batch_spec.size

        for itr in range(self.n_iterations):
            elapsed_steps = itr * batch_size

            # if self.eval_sampler is not None and itr % self.eval_interval_iters == 0:
            #     self.evaluate_agent(elapsed_steps)

            # logs at 0th iteration only if there is an eval sampler
            if itr % self.log_interval_iters == 0 and (
                itr > 0 or self.eval_sampler is not None
            ):
                self.log_progress(elapsed_steps, itr)

            batch_samples, completed_trajs = self.sampler.collect_batch(
                state, elapsed_steps
            )
            self.record_completed_trajectories(
                completed_trajs,
                prefix="sampling" if self.eval_sampler is not None else "trajectory",
            )

            algo_info = self.algorithm.optimize_agent(
                elapsed_steps,
                batch_samples,
            )
            self.record_algo_info(algo_info)

            progress_bar.update(batch_size)

        # log final progress
        elapsed_steps = self.n_iterations * batch_size
        if self.eval_sampler is not None:
            self.evaluate_agent(elapsed_steps)
        self.log_progress(self.n_iterations * batch_size, self.n_iterations)

        progress_bar.close()
        # TODO: replace with logger.finish method
        # logger.info(f"{type(self).__name__}: Finished training.")
        # if logger.log_dir is not None:
        #     logger.info(f"{type(self).__name__}: Log files saved to {logger.log_dir}")
