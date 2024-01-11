# fmt: off
from __future__ import annotations

import copy
from typing import TypedDict

import torch
from torch import Tensor

import parllel.logger as logger
from parllel import Array, ArrayDict, ArrayTree, Index, dict_map
from parllel.torch.distributions.squashed_gaussian import (DistParams,
                                                           SquashedGaussian)
from parllel.torch.utils import update_state_dict

from .agent import TorchAgent

# fmt: on
PiModelOutputs = DistParams


class QModelOutputs(TypedDict):
    q_value: Tensor


class SacAgent(TorchAgent):
    """Agent for SAC algorithm, including action-squashing, using twin Q-values."""

    model: torch.nn.ModuleDict
    distribution: SquashedGaussian

    def __init__(
        self,
        model: torch.nn.ModuleDict,
        distribution: SquashedGaussian,
        device: torch.device,
        learning_starts: int = 0,
        pretrain_std: float = 0.75,  # With squash 0.75 is near uniform.
    ) -> None:
        """Saves input arguments; network defaults stored within."""
        model["target_q1"] = copy.deepcopy(model["q1"])
        model["target_q1"].requires_grad_(False)
        model["target_q2"] = copy.deepcopy(model["q2"])
        model["target_q2"].requires_grad_(False)

        if "encoder" in model:
            model["target_encoder"] = copy.deepcopy(model["encoder"])
            model["target_encoder"].requires_grad_(False)

        super().__init__(model, distribution, device)

        self.learning_starts = learning_starts
        self.pretrain_std = pretrain_std

        self.recurrent = False

    def encode(self, observation: ArrayTree[Tensor]) -> ArrayTree[Tensor]:
        if "encoder" in self.model:
            observation = self.model["encoder"](observation)
        return observation

    def target_encode(self, observation: ArrayTree[Tensor]) -> ArrayTree[Tensor]:
        if "target_encoder" in self.model:
            observation = self.model["target_encoder"](observation)
        return observation

    @torch.no_grad()
    def step(
        self,
        observation: ArrayTree[Array],
        *,
        env_indices: Index = ...,
    ) -> tuple[Tensor, ArrayDict[Tensor]]:
        observation = observation.to_ndarray()
        observation = dict_map(torch.from_numpy, observation)
        observation = observation.to(device=self.device)
        encoding = self.encode(observation)
        dist_params: PiModelOutputs = self.model["pi"](encoding)
        action = self.distribution.sample(dist_params)
        return action.cpu(), ArrayDict()

    def q(
        self,
        observation: ArrayTree[Tensor],
        action: ArrayTree[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute twin Q-values for state/observation and input action
        (with grad)."""
        q1: QModelOutputs = self.model["q1"](observation, action)
        q2: QModelOutputs = self.model["q2"](observation, action)
        return q1["q_value"], q2["q_value"]

    def target_q(
        self,
        observation: ArrayTree[Tensor],
        action: ArrayTree[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute twin target Q-values for state/observation and input
        action."""
        target_q1: QModelOutputs = self.model["target_q1"](observation, action)
        target_q2: QModelOutputs = self.model["target_q2"](observation, action)
        return target_q1["q_value"], target_q2["q_value"]

    def pi(
        self,
        observation: ArrayTree[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        dist_params: PiModelOutputs = self.model["pi"](observation)
        action, log_pi = self.distribution.sample_loglikelihood(dist_params)
        return action, log_pi

    def freeze_q_models(self, freeze: bool) -> None:
        self.model["q1"].requires_grad_(not freeze)
        self.model["q2"].requires_grad_(not freeze)

    def update_target(self, tau: float | int = 1) -> None:
        update_state_dict(self.model["target_q1"], self.model["q1"].state_dict(), tau)
        update_state_dict(self.model["target_q2"], self.model["q2"].state_dict(), tau)
        if "target_encoder" in self.model:
            update_state_dict(
                self.model["target_encoder"], self.model["encoder"].state_dict(), tau
            )

    def train_mode(self, elapsed_steps: int) -> None:
        super().train_mode(elapsed_steps)
        self.distribution.set_fixed_std(None)

    def sample_mode(self, elapsed_steps: int) -> None:
        super().sample_mode(elapsed_steps)
        if elapsed_steps == 0 and self.learning_starts > 0:
            logger.info(
                f"For the first {self.learning_starts} steps, agent will use a fixed std of {self.pretrain_std} for exploration."
            )
        std = None if elapsed_steps >= self.learning_starts else self.pretrain_std
        self.distribution.set_fixed_std(std)  # If None: std from policy dist_params.

    def eval_mode(self, elapsed_steps: int) -> None:
        super().eval_mode(elapsed_steps)
        self.distribution.set_fixed_std(None)
