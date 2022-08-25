from pathlib import Path
from typing import Optional

from tqdm import tqdm

from parllel.algorithm import Algorithm
from parllel.samplers import Sampler, EvalSampler
from parllel.handlers import Agent
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
        log_dir: Optional[Path] = None,
    ) -> None:

        super().__init__(
            log_dir=log_dir,
        )
    
        self.sampler = sampler
        self.agent = agent
        self.algorithm = algorithm
        self.batch_spec = batch_spec
        self.eval_sampler = eval_sampler
        self.n_steps = n_steps

        self.n_iterations = int(n_steps // batch_spec.size)
        self.log_interval_iters = int(log_interval_steps // batch_spec.size)

    def run(self) -> None:
        print("Starting training...")
        
        progress_bar = tqdm(total=self.n_steps, unit="steps")
        batch_size = self.batch_spec.size

        self._evaluate_agent(elapsed_steps=0)
        for itr in range(self.n_iterations):
            elapsed_steps = itr * batch_size

            batch_samples, _ = self.sampler.collect_batch(elapsed_steps)

            self.algorithm.optimize_agent(elapsed_steps, batch_samples)

            if (itr + 1) % self.log_interval_iters == 0:
                self._evaluate_agent(elapsed_steps=elapsed_steps)

            progress_bar.update(batch_size)
        
        print("Finished training.")
        progress_bar = None
        
    def _evaluate_agent(self, elapsed_steps: int) -> None:
        eval_trajs = self.eval_sampler.collect_batch(elapsed_steps)
        self.log_progress(elapsed_steps, eval_trajs)
