from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    has_summary_writer = True
except ImportError:
    has_summary_writer = False

from parllel.algorithm import Algorithm
from parllel.handlers.agent import Agent
from parllel.samplers.sampler import Sampler
from parllel.types import BatchSpec


class OnPolicyRunner:
    def __init__(self,
            sampler: Sampler,
            agent: Agent,
            algorithm: Algorithm,
            n_steps: int,
            batch_spec: BatchSpec,
            log_interval_steps: int,
            log_dir: Optional[Path] = None,
        ) -> None:
        self.sampler = sampler
        self.agent = agent
        self.algorithm = algorithm
        self.n_steps = n_steps
        self.batch_spec = batch_spec

        if log_dir is not None and has_summary_writer:
            log_dir.mkdir(parents=True)
            self.logger = SummaryWriter(log_dir=str(log_dir))
        else:
            self.logger = None

        self.n_iterations = int(n_steps // batch_spec.size)
        self.log_interval_iters = int(log_interval_steps // batch_spec.size)

        self._progress_bar = None

    def run(self) -> None:
        print("Starting training...")
        
        self._progress_bar = tqdm(total=self.n_steps, unit="steps")
        batch_size = self.batch_spec.size
        completed_trajs = []

        for itr in range(self.n_iterations):
            elapsed_steps = itr * batch_size

            batch_samples, new_trajs = self.sampler.collect_batch(elapsed_steps)
            completed_trajs.extend(new_trajs)

            self.algorithm.optimize_agent(elapsed_steps, batch_samples)

            if (itr + 1) % self.log_interval_iters == 0:
                traj_disc_returns = [traj.DiscountedReturn for traj in completed_trajs]
                traj_disc_returns = np.array(traj_disc_returns)
                mean_disc_return = traj_disc_returns.mean()

                if self.logger is not None:
                    self.logger.add_scalar("DiscountedReturn", mean_disc_return, elapsed_steps)

                print(f"Average discounted return: {mean_disc_return:.3f}")

                completed_trajs.clear()

            self._progress_bar.update(batch_size)

        print("Finished training.")
        self._progress_bar = None
