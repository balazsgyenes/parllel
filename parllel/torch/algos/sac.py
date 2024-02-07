from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import parllel.logger as logger
import torch
from gymnasium import spaces
from parllel import Array, ArrayDict
from parllel.algorithm import Algorithm
from parllel.replays.replay import ReplayBuffer
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.utils import valid_mean
from parllel.types.batch_spec import BatchSpec
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler


class SAC(Algorithm):
    """Soft actor critic algorithm, training from a replay buffer."""

    def __init__(
        self,
        batch_spec: BatchSpec,
        agent: SacAgent,
        replay_buffer: ReplayBuffer[ArrayDict[Tensor]],
        q_optimizer: torch.optim.Optimizer,
        pi_optimizer: torch.optim.Optimizer,
        discount: float,
        learning_starts: int,
        replay_ratio: int,  # data_consumption / data_generation
        target_update_tau: float,  # tau=1 for hard update.
        target_update_interval: int,  # 1000 for hard update, 1 for soft.
        ent_coeff: float,
        ent_coeff_lr: float | None = None,
        action_space: spaces.Box | None = None,
        clip_grad_norm: float | None = None,
        learning_rate_schedulers: Sequence[LRScheduler] | None = None,
        **kwargs,  # ignore additional arguments
    ):
        """Save input arguments."""
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.q_optimizer = q_optimizer
        self.pi_optimizer = pi_optimizer
        self.discount = discount
        self.learning_starts = int(learning_starts)
        self.replay_ratio = replay_ratio
        self.target_update_tau = target_update_tau
        self.target_update_interval = target_update_interval
        self.clip_grad_norm = clip_grad_norm
        self.lr_schedulers = learning_rate_schedulers

        logger.debug(
            f"{type(self).__name__}: Ignoring remaining kwargs: "
            f"{','.join([f'{key}={value}' for key, value in kwargs.items()])}"
        )

        replay_batch_size = self.replay_buffer.replay_batch_size
        self.updates_per_optimize = int(
            max(1, self.replay_ratio * batch_spec.size / replay_batch_size)
        )
        logger.info(
            f"{type(self).__name__}: Given sampler batch size {batch_spec.size}, "
            f"training batch size {replay_batch_size}, and replay ratio "
            f"{self.replay_ratio}, there will be {self.updates_per_optimize} "
            f"updates per iteration."
        )
        self.update_counter = 0
        self.algo_log_info = defaultdict(list)

        if ent_coeff_lr is not None:
            assert action_space is not None
            init_value = ent_coeff
            self.target_entropy = -np.prod(action_space.shape).item()  # type: ignore
            self._log_ent_coeff = (
                torch.log(torch.tensor(init_value))
                .to(device=agent.device)
                .requires_grad_(True)
            )
            self.ent_coeff_optimizer = torch.optim.Adam(
                [self._log_ent_coeff], lr=ent_coeff_lr
            )
            logger.info(
                f"{type(self).__name__}: Using learnable entropy coefficient with target entropy of {self.target_entropy}"
            )
        else:
            self._ent_coeff = torch.tensor([ent_coeff]).to(agent.device)
            self._log_ent_coeff = torch.log(self._ent_coeff).to(agent.device)
            self.ent_coeff_optimizer = None

    def optimize_agent(
        self,
        elapsed_steps: int,
        samples: ArrayDict[Array],
    ) -> dict[str, int | list[float]]:
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

        if self.lr_schedulers is not None:
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step()
            # fmt: off
            self.algo_log_info["pi_learning_rate"] = self.pi_optimizer.param_groups[0]["lr"]
            self.algo_log_info["q_learning_rate"] = self.q_optimizer.param_groups[0]["lr"]
            # fmt: on

        self.algo_log_info["n_updates"] = self.update_counter

        return self.algo_log_info

    def train_once(self, samples: ArrayDict[Tensor]) -> None:

        q_loss = self.critic_loss(samples)

        # update Q model parameters according to Q loss
        self.q_optimizer.zero_grad()
        q_loss.backward()

        if self.clip_grad_norm is not None:
            # clip all gradients except for pi, which is not updated yet
            for key, model in self.agent.model.items():
                if "target" not in key and key != "pi":
                    q_grad_norm = clip_grad_norm_(
                        model.parameters(), self.clip_grad_norm
                    )
                    self.algo_log_info[f"{key}_grad_norm"].append(q_grad_norm.item())

        self.q_optimizer.step()

        # freeze Q models while optimizing policy model
        self.agent.freeze_q_models(True)

        pi_loss, ent_coeff_loss = self.actor_loss(samples)

        # unfreeze Q models for next training iteration
        self.agent.freeze_q_models(False)

        # update Pi model parameters according to pi loss
        self.pi_optimizer.zero_grad()
        pi_loss.backward()

        if self.clip_grad_norm is not None:
            pi_grad_norm = clip_grad_norm_(
                self.agent.model["pi"].parameters(),
                self.clip_grad_norm,
            )
            self.algo_log_info["pi_grad_norm"].append(pi_grad_norm.item())

        self.pi_optimizer.step()

        if self.ent_coeff_optimizer is not None:
            assert ent_coeff_loss is not None
            self.ent_coeff_optimizer.zero_grad()
            ent_coeff_loss.backward()
            self.ent_coeff_optimizer.step()

    def critic_loss(self, samples: ArrayDict[Tensor]) -> Tensor:
        """
        Computes losses for twin Q-values against the min of twin target Q-values
        and an entropy term.  Computes reparameterized policy loss, and loss for
        tuning entropy weighting, alpha. If an encoder is present, it is trained
        only on the critic loss.

        Input samples have leading batch dimension [B,..] (but not time).
        """
        entropy_coeff = (
            torch.exp(self._log_ent_coeff.detach())
            if self.ent_coeff_optimizer is not None
            else self._ent_coeff
        )

        # compute target Q according to formula
        # r + gamma * (1 - d) * (min Q_targ(s', a') - alpha * log pi(s', a'))
        # where a' ~ pi(.|s')
        with torch.no_grad():
            # encode for learning policy network with learning encoder
            next_observation = self.agent.encode(samples["next_observation"])
            next_action, next_log_prob = self.agent.pi(next_observation)

            # encode for target_q network with target encoder
            next_observation = self.agent.target_encode(samples["next_observation"])
            target_q1, target_q2 = self.agent.target_q(next_observation, next_action)

            min_target_q = torch.min(target_q1, target_q2)
            entropy_bonus = -entropy_coeff * next_log_prob
            y = samples["reward"] + self.discount * ~samples["terminated"] * (
                min_target_q + entropy_bonus
            )

        # encode once and add it back to the samples, allowing the actor loss
        # to reuse the value
        observation = self.agent.encode(samples["observation"])
        samples["encoding"] = observation

        q1, q2 = self.agent.q(observation, samples["action"])
        q_loss = 0.5 * valid_mean((y - q1) ** 2 + (y - q2) ** 2)

        self.algo_log_info["critic_loss"].append(q_loss.item())
        self.algo_log_info["mean_entropy"].append(-next_log_prob.mean().item())
        self.algo_log_info["mean_ent_bonus"].append(entropy_bonus.mean().item())
        self.algo_log_info["max_target_q"].append(min_target_q.max().item())
        self.algo_log_info["min_target_q"].append(min_target_q.min().item())
        self.algo_log_info["max_reward"].append(samples["reward"].max().item())
        self.algo_log_info["min_reward"].append(samples["reward"].min().item())

        return q_loss

    def actor_loss(self, samples: ArrayDict[Tensor]) -> tuple[Tensor, Tensor | None]:
        """Compute reparametrized policy loss by maximizing the predicted Q value.
        maximize (min Q(s, a) - alpha * log pi(a, s))
        where a ~ pi(.|s)
        """
        entropy_coeff = (
            torch.exp(self._log_ent_coeff.detach())
            if self.ent_coeff_optimizer is not None
            else self._ent_coeff
        )

        # detach the encoded observation from the encoder gradients
        # the encoder was already updated using the critic loss
        observation = samples["encoding"].detach()

        new_action, log_prob = self.agent.pi(observation)
        q1, q2 = self.agent.q(observation.detach(), new_action)
        min_q = torch.min(q1, q2)
        pi_losses = entropy_coeff * log_prob - min_q
        pi_loss = valid_mean(pi_losses)

        self.algo_log_info["actor_loss"].append(pi_loss.item())

        if self.ent_coeff_optimizer is not None:
            self.algo_log_info["ent_coeff"].append(entropy_coeff.item())
            ent_coeff_loss = -(
                self._log_ent_coeff * (log_prob.detach().mean() + self.target_entropy)
            )
            self.algo_log_info["ent_coeff_loss"].append(ent_coeff_loss.item())
        else:
            ent_coeff_loss = None

        return pi_loss, ent_coeff_loss


def build_replay_buffer_tree(sample_buffer: ArrayDict[Array]) -> ArrayDict[Array]:
    replay_buffer_tree = ArrayDict(
        {
            "observation": sample_buffer["observation"].full,
            "action": sample_buffer["action"].full,
            "reward": sample_buffer["reward"].full,
            "terminated": sample_buffer["terminated"].full,
            "next_observation": sample_buffer["next_observation"].full,
        }
    )
    return replay_buffer_tree
