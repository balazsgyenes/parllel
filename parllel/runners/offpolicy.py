import numpy as np
from tqdm import tqdm
from parllel.algorithm import Algorithm

from parllel.samplers.sampler import Sampler
from parllel.handlers.agent import Agent
from parllel.types import BatchSpec


class OffPolicyRunner:
    def __init__(self,
            sampler: Sampler,
            agent: Agent,
            algorithm: Algorithm,
            batch_spec: BatchSpec,
            n_steps: int,
            log_interval_steps: int,
        ) -> None:
        self.sampler = sampler
        self.agent = agent
        self.algorithm = algorithm


        self.n_steps = n_steps
        self.batch_spec = batch_spec

        self.n_iterations = int(n_steps // batch_spec.size)
        self.log_interval_iters = int(log_interval_steps // batch_spec.size)

    def run(self) -> None:
        progress_bar = tqdm(total=self.n_steps, unit="steps")
        batch_size = self.batch_spec.size

        self._evaluate_agent(0)
        for itr in range(self.n_iterations):
            elapsed_steps = itr * batch_size

            batch_samples, completed_trajs = self.sampler.collect_batch(elapsed_steps)

            self.algorithm.optimize_agent(elapsed_steps, batch_samples)

            if (itr + 1) % self.log_interval_iters == 0:
                self._evaluate_agent(elapsed_steps)

            progress_bar.update(batch_size)
        
        print("Finished training.")
        
    def _evaluate_agent(self, elapsed_steps):
        self.agent.eval_mode(elapsed_steps)
        # TODO: this is super dirty, don't do this
        batch_samples, completed_trajs = self.sampler.collect_batch(elapsed_steps)
        traj_disc_returns = [traj.DiscountedReturn for traj in completed_trajs]
        traj_disc_returns = np.array(traj_disc_returns)
        print(f"Average discounted return: {traj_disc_returns.mean():.3f}")