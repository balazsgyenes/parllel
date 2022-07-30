from typing import Dict, Union

import torch

from parllel.algorithm import Algorithm
from parllel.arrays import Array
from parllel.buffers import Samples
from parllel.replays.replay import ReplayBuffer
from parllel.torch.agents.sac_agent import SacAgent
from parllel.types.batch_spec import BatchSpec
from parllel.torch.utils import buffer_to_device, torchify_buffer, valid_mean


class SAC(Algorithm):
    """Soft actor critic algorithm, training from a replay buffer."""

    def __init__(self,
        batch_spec: BatchSpec,
        agent: SacAgent,
        replay_buffer: ReplayBuffer,
        optimizers: Dict[str, torch.optim.Optimizer],
        batch_size: int = 256,
        discount: float = 0.99,
        learning_starts: int = 0,
        replay_ratio=256,  # data_consumption / data_generation
        target_update_tau=0.005,  # tau=1 for hard update.
        target_update_interval=1,  # 1000 for hard update, 1 for soft.
        ent_coeff: Union[str, float] = 1e-5, # "auto" for adaptive alpha, float for any fixed value
        clip_grad_norm: float = 1e9,
    ):
        """Save input arguments."""
        self.batch_spec = batch_spec
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.optimizers = optimizers
        self.batch_size = batch_size
        self.discount = discount
        self.learning_starts = learning_starts
        self.replay_ratio = replay_ratio
        self.target_update_tau = target_update_tau 
        self.target_update_interval = target_update_interval
        self.clip_grad_norm = clip_grad_norm

        self.updates_per_optimize = int(self.replay_ratio * self.batch_spec.size /
            self.batch_size)
        print(f"From sampler batch size {self.batch_spec.size}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration.")
        self.update_counter = 0

        self._alpha = torch.tensor([ent_coeff]).to(agent.device)
        self._log_alpha = torch.log(self._alpha).to(agent.device)

    def optimize_agent(self, elapsed_steps: int, samples: Samples[Array]):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """
        self.replay_buffer.append_samples(samples)
        
        if elapsed_steps < self.learning_starts:
            return
        
        self.agent.train_mode(elapsed_steps)
        
        for _ in range(self.updates_per_optimize):

            self.train_once(elapsed_steps)            

            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)

    def train_once(self, elapsed_steps: int):
        """
        Computes losses for twin Q-values against the min of twin target Q-values
        and an entropy term.  Computes reparameterized policy loss, and loss for
        tuning entropy weighting, alpha.  
        
        Input samples have leading batch dimension [B,..] (but not time).
        """
        # get a random batch of samples from the replay buffer and move them
        # to the GPU
        samples = self.replay_buffer.sample_batch(self.batch_size)
        samples = torchify_buffer(samples)
        samples = buffer_to_device(samples, self.agent.device)

        # compute target Q according to formula
        # r + gamma * (1 - d) * (min Q_targ(s', a') - alpha * log pi(s', a'))
        # where a' ~ pi(.|s')
        with torch.no_grad():
            next_action, next_log_prob, _ = self.agent.pi(samples.next_observation)
            target_q1, target_q2 = self.agent.target_q(samples.next_observation, next_action)
        min_target_q = torch.min(target_q1, target_q2)
        next_q = min_target_q - self._alpha * next_log_prob
        y = samples.reward + self.discount * ~samples.done * next_q
        q1, q2 = self.agent.q(samples.observation, samples.action)
        q_loss = 0.5 * valid_mean((y - q1) ** 2 + (y - q2) ** 2)

        # update Q model parameters according to Q loss
        self.optimizers["q"].zero_grad()
        q_loss.backward()
        q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.model["q1"].parameters(),
            self.clip_grad_norm)
        q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.model["q2"].parameters(),
            self.clip_grad_norm)
        self.optimizers["q"].step()

        # freeze Q models while optimizing policy model
        self.agent.freeze_q_models(True)

        # train policy model by maximizing the predicted Q value
        # maximize (min Q(s, a) - alpha * log pi(a, s))
        # where a ~ pi(.|s)
        new_action, log_prob, (pi_mean, pi_log_std) = self.agent.pi(samples.observation)
        q1, q2 = self.agent.q(samples.observation, new_action)
        min_q = torch.min(q1, q2)
        pi_losses = self._alpha * log_prob - min_q
        pi_loss = valid_mean(pi_losses)

        # update Pi model parameters according to pi loss
        self.optimizers["pi"].zero_grad()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.model["pi"].parameters(),
            self.clip_grad_norm)
        self.optimizers["pi"].step()

        # unfreeze Q models for next training iteration
        self.agent.freeze_q_models(False)
