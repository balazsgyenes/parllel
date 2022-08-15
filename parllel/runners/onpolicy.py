from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    has_summary_writer = True
except ImportError:
    has_summary_writer = False

from parllel.algorithm import Algorithm
from parllel.cages.traj_info import TrajInfo
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

        self.log_dir = None
        self.logger = None
        if log_dir is not None:
            print(f"Saving model checkpoints to {log_dir}.")
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if has_summary_writer:
                print("Saving learning statistics to tensorboard.")
                self.logger = SummaryWriter(log_dir=str(log_dir))

        self.n_iterations = int(n_steps // batch_spec.size)
        self.log_interval_iters = int(log_interval_steps // batch_spec.size)

    def run(self) -> None:
        print("Starting training...")
        
        progress_bar = tqdm(total=self.n_steps, unit="steps")
        batch_size = self.batch_spec.size
        completed_trajs = []

        for itr in range(self.n_iterations):
            elapsed_steps = itr * batch_size

            batch_samples, new_trajs = self.sampler.collect_batch(elapsed_steps)
            completed_trajs.extend(new_trajs)

            self.algorithm.optimize_agent(elapsed_steps, batch_samples)

            if (itr + 1) % self.log_interval_iters == 0:
                self.log_progress(elapsed_steps, completed_trajs)
                completed_trajs.clear()

            progress_bar.update(batch_size)

        print("Finished training.")
        progress_bar = None

    def log_progress(self, elapsed_steps: int, trajectories: List[TrajInfo]) -> None:
        traj_disc_returns = [traj.DiscountedReturn for traj in trajectories]
        traj_disc_returns = np.array(traj_disc_returns)
        mean_disc_return = traj_disc_returns.mean()

        if self.logger is not None:
            self.logger.add_scalar("DiscountedReturn", mean_disc_return, elapsed_steps)

        if self.log_dir is not None:
            self.agent.save_model(path=self.log_dir / "model.pt")

        print(f"Average discounted return: {mean_disc_return:.3f}")
