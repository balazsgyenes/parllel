from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_

import parllel.logger as logger
from parllel import Array, ArrayDict
from parllel.algorithm import Algorithm
from parllel.replays.replay import ReplayBuffer
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.utils import valid_mean
from parllel.types.batch_spec import BatchSpec


class SAC(Algorithm):
    """Soft actor critic algorithm, training from a replay buffer."""

    def __init__(
        self,
        batch_spec: BatchSpec,
        agent: SacAgent,
        replay_buffer: ReplayBuffer[ArrayDict[Tensor]],
        optimizers: Mapping[str, torch.optim.Optimizer],
        discount: float,
        learning_starts: int,
        replay_ratio: int,  # data_consumption / data_generation
        target_update_tau: float,  # tau=1 for hard update.
        target_update_interval: int,  # 1000 for hard update, 1 for soft.
        ent_coeff: float,
        clip_grad_norm: float | None = None,
        **kwargs,  # ignore additional arguments
    ):
        """Save input arguments."""
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.optimizers = optimizers
        self.discount = discount
        self.learning_starts = int(learning_starts)
        self.replay_ratio = replay_ratio
        self.target_update_tau = target_update_tau
        self.target_update_interval = target_update_interval
        self.clip_grad_norm = clip_grad_norm

        replay_batch_size = self.replay_buffer.replay_batch_size
        self.updates_per_optimize = int(
            self.replay_ratio * batch_spec.size / replay_batch_size
        )
        logger.info(
            f"{type(self).__name__}: From sampler batch size {batch_spec.size}, "
            f"training batch size {replay_batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration."
        )
        self.update_counter = 0
        self.algo_log_info = defaultdict(list)

        self._alpha = torch.tensor([ent_coeff]).to(agent.device)
        self._log_alpha = torch.log(self._alpha).to(agent.device)

    def optimize_agent(
        self,
        elapsed_steps: int,
        samples: ArrayDict[Array],
    ) -> dict[str, int | list[float]]:
        """
        Extracts the needed fields from input samples and stores them in the
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """
        self.replay_buffer.next_iteration()

        if elapsed_steps < self.learning_starts:
            logger.debug(
                f"Skipping optimization at {elapsed_steps} steps, waiting until {self.learning_starts} steps."
            )
            return {}

        self.agent.train_mode(elapsed_steps)
        self.algo_log_info.clear()

        for _ in range(self.updates_per_optimize):
            # get a random batch of samples from the replay buffer and move them
            # to the GPU
            replay_samples = self.replay_buffer.sample_batch()

            self.train_once(replay_samples)

            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)

            self.algo_log_info["n_updates"] = self.update_counter

        return self.algo_log_info

    def train_once(self, samples: ArrayDict[Tensor]) -> None:
        """
        Computes losses for twin Q-values against the min of twin target Q-values
        and an entropy term.  Computes reparameterized policy loss, and loss for
        tuning entropy weighting, alpha.

        Input samples have leading batch dimension [B,..] (but not time).
        """
        # encode once, allowing the agent to reuse its encodings for q and
        # pi predictions.
        # we then later detach the encoding from its computational graph
        # during q prediction, since we only want to optimize the encoder
        # using the gradients from the policy network.
        observation = self.agent.encode(samples["observation"])

        # compute target Q according to formula
        # r + gamma * (1 - d) * (min Q_targ(s', a') - alpha * log pi(s', a'))
        # where a' ~ pi(.|s')
        with torch.no_grad():
            next_observation = self.agent.encode(samples["next_observation"])
            next_action, next_log_prob = self.agent.pi(next_observation)
            target_q1, target_q2 = self.agent.target_q(next_observation, next_action)
        min_target_q = torch.min(target_q1, target_q2)
        next_q = min_target_q - self._alpha * next_log_prob
        y = samples["reward"] + self.discount * ~samples["done"] * next_q
        q1, q2 = self.agent.q(observation.detach(), samples["action"])
        q_loss = 0.5 * valid_mean((y - q1) ** 2 + (y - q2) ** 2)
        self.algo_log_info["critic_loss"].append(q_loss.item())

        # update Q model parameters according to Q loss
        self.optimizers["q"].zero_grad()
        q_loss.backward()

        if self.clip_grad_norm is not None:
            q1_grad_norm = clip_grad_norm_(
                self.agent.model["q1"].parameters(),
                self.clip_grad_norm,
            )
            q2_grad_norm = clip_grad_norm_(
                self.agent.model["q2"].parameters(),
                self.clip_grad_norm,
            )
            self.algo_log_info["q1_grad_norm"].append(q1_grad_norm.item())
            self.algo_log_info["q2_grad_norm"].append(q2_grad_norm.item())

        self.optimizers["q"].step()

        # freeze Q models while optimizing policy model
        self.agent.freeze_q_models(True)

        # train policy model by maximizing the predicted Q value
        # maximize (min Q(s, a) - alpha * log pi(a, s))
        # where a ~ pi(.|s)
        new_action, log_prob = self.agent.pi(observation)
        q1, q2 = self.agent.q(observation.detach(), new_action)
        min_q = torch.min(q1, q2)
        pi_losses = self._alpha * log_prob - min_q
        pi_loss = valid_mean(pi_losses)
        self.algo_log_info["actor_loss"].append(pi_loss.item())

        # update Pi model parameters according to pi loss
        self.optimizers["pi"].zero_grad()
        pi_loss.backward()

        if self.clip_grad_norm is not None:
            pi_grad_norm = clip_grad_norm_(
                self.agent.model["pi"].parameters(),
                self.clip_grad_norm,
            )
            self.algo_log_info["pi_grad_norm"].append(pi_grad_norm.item())

        self.optimizers["pi"].step()

        # unfreeze Q models for next training iteration
        self.agent.freeze_q_models(False)

        self.algo_log_info["mean_min_q"].append(min_q.detach().mean().item())
        self.algo_log_info["max_min_q"].append(min_q.detach().max().item())
        self.algo_log_info["min_min_q"].append(min_q.detach().min().item())


def build_replay_buffer_tree(sample_buffer: ArrayDict[Array]) -> ArrayDict[Array]:
    replay_buffer_tree = ArrayDict(
        {
            "observation": sample_buffer["observation"].full,
            "action": sample_buffer["action"].full,
            "reward": sample_buffer["reward"].full,
            "terminated": sample_buffer["terminated"].full,
            "truncated": sample_buffer["truncated"].full,
            "done": sample_buffer["done"].full,
            "next_observation": sample_buffer["observation"].full.next,
        }
    )
    return replay_buffer_tree
