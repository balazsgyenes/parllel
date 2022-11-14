import copy
from dataclasses import dataclass
from typing import Tuple, Union

import gym
import torch

from parllel.buffers import Buffer, NamedArrayTupleClass
from parllel.handlers import AgentStep
import parllel.logger as logger
from parllel.torch.agents.agent import TorchAgent
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian, DistInfoStd
from parllel.torch.utils import buffer_to_device, update_state_dict


AgentInfo = NamedArrayTupleClass("AgentInfo", ["dist_info"])

@dataclass(frozen=True)
class PiModelOutputs:
    mean: Buffer
    log_std: Buffer

@dataclass(frozen=True)
class QModelOutputs:
    q_value: Buffer


class SacAgent(TorchAgent):
    """Agent for SAC algorithm, including action-squashing, using twin Q-values."""

    def __init__(self,
        model: torch.nn.Module,
        distribution: SquashedGaussian,
        device: torch.device,
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_starts: int = 0,
        pretrain_std: float = 0.75,  # With squash 0.75 is near uniform.
    ) -> None:
        """Saves input arguments; network defaults stored within."""
        model["target_q1"] = copy.deepcopy(model["q1"])
        model["target_q1"].requires_grad_(False)
        model["target_q2"] = copy.deepcopy(model["q2"])
        model["target_q2"].requires_grad_(False)

        super().__init__(model, distribution, device)

        self.obs_space = observation_space
        self.action_space = action_space
        self.learning_starts = learning_starts
        self.pretrain_std = pretrain_std

        self.recurrent = False

    @torch.no_grad()
    def step(self,
        observation: Buffer[torch.Tensor],
        *,
        env_indices: Union[int, slice] = ...,
    ) -> AgentStep:
        model_inputs = (observation,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        model_outputs: PiModelOutputs = self.model["pi"](*model_inputs)

        dist_info = DistInfoStd(mean=model_outputs.mean,
                                log_std=model_outputs.log_std)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info)
        action, agent_info = buffer_to_device((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def q(self,
        observation: Buffer[torch.Tensor],
        action: Buffer[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute twin Q-values for state/observation and input action 
        (with grad)."""
        model_inputs = (observation, action)
        q1: QModelOutputs = self.model["q1"](*model_inputs)
        q2: QModelOutputs = self.model["q2"](*model_inputs)
        return q1.q_value, q2.q_value

    def target_q(self,
        observation: Buffer[torch.Tensor],
        action: Buffer[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute twin target Q-values for state/observation and input
        action.""" 
        model_inputs = (observation, action)
        target_q1: QModelOutputs = self.model["target_q1"](*model_inputs)
        target_q2: QModelOutputs = self.model["target_q2"](*model_inputs)
        return target_q1.q_value, target_q2.q_value

    def pi(self,
        observation: Buffer[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, DistInfoStd]:
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_inputs = (observation,)
        model_outputs: PiModelOutputs = self.model["pi"](*model_inputs)
        dist_info = DistInfoStd(mean=model_outputs.mean, log_std=model_outputs.log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        return action, log_pi, dist_info

    def freeze_q_models(self, freeze: bool) -> None:
        self.model["q1"].requires_grad_(not freeze)
        self.model["q2"].requires_grad_(not freeze)

    def update_target(self, tau: Union[float, int] = 1) -> None:
        update_state_dict(self.model["target_q1"], self.model["q1"].state_dict(), tau)
        update_state_dict(self.model["target_q2"], self.model["q2"].state_dict(), tau)

    def train_mode(self, elapsed_steps: int) -> None:
        super().train_mode(elapsed_steps)
        self.model["q1"].train()
        self.model["q2"].train()

    def sample_mode(self, elapsed_steps: int) -> None:
        super().sample_mode(elapsed_steps)
        self.model["q1"].eval()
        self.model["q2"].eval()
        if elapsed_steps == 0:
            logger.debug(f"Agent at {elapsed_steps} steps, sample std: {self.pretrain_std}")
        std = None if elapsed_steps >= self.learning_starts else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, elapsed_steps: int) -> None:
        super().eval_mode(elapsed_steps)
        self.model["q1"].eval()
        self.model["q2"].eval()
        self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).
