from typing import Dict, Union

import torch

from parllel.algorithm import Algorithm
from parllel.buffers import Samples
from parllel.replays.replay import ReplayBuffer
from parllel.torch.agents.sac_agent import SacAgent
from parllel.types.batch_spec import BatchSpec
from parllel.torch.utils import buffer_to_device, torchify_buffer, valid_mean


class SAC(Algorithm):
    """Soft actor critic algorithm, training from a replay buffer."""

    def __init__(
            self,
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
            clip_grad_norm=1e9,
            ):
        """Save input arguments."""
        self.batch_spec = batch_spec
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.pi_optimizer = optimizers["pi"]
        self.q_optimizer = optimizers["q"]
        self.alpha_optimizer = None
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

    def optimize_agent(self, elapsed_steps: int, samples: Samples):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """
        self.replay_buffer.append_samples(samples)
        # opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if elapsed_steps < self.learning_starts:
            return
        for _ in range(self.updates_per_optimize):
            self.agent.train_mode(elapsed_steps)

            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            samples_from_replay = torchify_buffer(samples_from_replay)
            samples_from_replay = buffer_to_device(samples_from_replay, self.agent.device)

            losses, values = self.loss(samples_from_replay)
            q_loss, pi_loss, alpha_loss = losses

            if alpha_loss is not None:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self._alpha = torch.exp(self._log_alpha.detach())

            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.model["pi"].parameters(),
                self.clip_grad_norm)
            self.pi_optimizer.step()

            self.q_optimizer.zero_grad()
            q_loss.backward()
            q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.model["q1"].parameters(),
                self.clip_grad_norm)
            q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.model["q2"].parameters(),
                self.clip_grad_norm)
            self.q_optimizer.step()

            grad_norms = (q1_grad_norm, q2_grad_norm, pi_grad_norm)

            # self.append_opt_info_(opt_info, losses, grad_norms, values)
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)

    def loss(self, samples):
        """
        Computes losses for twin Q-values against the min of twin target Q-values
        and an entropy term.  Computes reparameterized policy loss, and loss for
        tuning entropy weighting, alpha.  
        
        Input samples have leading batch dimension [B,..] (but not time).
        """
        q1, q2 = self.agent.q(samples.observation, samples.action)
        with torch.no_grad():
            target_action, target_log_pi, _ = self.agent.pi(samples.next_observation)
            target_q1, target_q2 = self.agent.target_q(samples.next_observation, target_action)
        min_target_q = torch.min(target_q1, target_q2)
        target_value = min_target_q - self._alpha * target_log_pi
        y = samples.reward + (1 - samples.done.float()) * self.discount * target_value

        q_loss = 0.5 * valid_mean((y - q1) ** 2 + (y - q2) ** 2)

        new_action, log_pi, (pi_mean, pi_log_std) = self.agent.pi(samples.observation)
        log_target1, log_target2 = self.agent.q(samples.observation, new_action)
        min_log_target = torch.min(log_target1, log_target2)

        pi_losses = self._alpha * log_pi - min_log_target
        pi_loss = valid_mean(pi_losses)

        alpha_loss = None

        losses = (q_loss, pi_loss, alpha_loss)
        values = tuple(val.detach() for val in (q1, q2, pi_mean, pi_log_std))
        return losses, values
