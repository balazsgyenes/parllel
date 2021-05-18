from dataclasses import dataclass
from typing import Any, Tuple

import torch
from nptyping import NDArray

from parllel.torch.agents.agent import AgentStep, Agent
from parllel.torch.agents.pg.types import AgentInfo, AgentEvaluation
from parllel.torch.distributions.categorical import Categorical, DistInfo
from parllel.torch.utils.buffer import buffer_to, buffer_func, buffer_method


@dataclass(frozen=True)
class ModelOutputs:
    pi: Any
    value: Any = None
    next_rnn_state: Any = None


class CategoricalPgAgent(Agent):
    """
    Agent for policy gradient algorithm using categorical action distribution.
    Same as ``GausssianPgAgent`` and related classes, except uses
    ``Categorical`` distribution, and has a different interface to the model
    (model here outputs discrete probabilities in place of means and log_stds,
    while both output the value estimate).
    """
    def __init__(self, recurrent: bool = True, actor_only: bool = False) -> None:
        self.recurrent = recurrent
        self.actor_only = actor_only  # TODO: where is this needed?

    def initialize(self, model: torch.Module, device: torch.device, distribution: Categorical, example_inputs: Tuple[NDArray, NDArray, NDArray], n_states: int) -> None:
        super().initialize(model, device, distribution)

        observation, prev_action, prev_reward = example_inputs
        prev_action = self._distribution.to_onehot(prev_action)
        model_inputs = buffer_to(
            (observation, prev_action, prev_reward), device=self.device
        )
        if self.recurrent:  # TODO: should we define that all models take 4 parameters?
            model_inputs += (None,)

        model_outputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        action = self.distribution.sample(dist_info)
        value = model_outputs.value
        rnn_state = model_outputs.next_rnn_state
        if rnn_state is not None:
            
            def extend_rnn_state_component(rnn_state_component):
                """Extend an rnn_state_component to allocate enough space for
                each env.
                """
                # duplicate as many times as requested
                rnn_state_components = (rnn_state_component) * n_states
                # concatenate in B dimension (shape should be [N,B,H])
                return torch.cat(rnn_state_components, dim=1)

            self._rnn_states = buffer_func(rnn_state, extend_rnn_state_component)

            # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
            rnn_state = buffer_method(rnn_state, "transpose", 0, 1)

        agent_info = AgentInfo(dist_info=dist_info, value=value, prev_rnn_state=rnn_state)
        agent_step = AgentStep(action=action, agent_info=agent_info)
        return buffer_to(agent_step, device="cpu")

    def evaluate(self, observation, prev_action, prev_reward, init_rnn_state=None):
        """Performs forward pass on training data, for algorithm."""
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = (observation, prev_action, prev_reward)
        if self.recurrent:
            model_inputs += (init_rnn_state,)
        model_inputs = buffer_to(model_inputs, device=self.device)
        model_outputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        output = (dist_info,)
        if not self.actor_only:
            output += (model_outputs.value,)
        output = buffer_to(output, device="cpu")
        if self.recurrent:  # Leave rnn_state on device
            output += (model_outputs.next_rnn_state,)
        return self.OutputCls(*output)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to(
            (observation, prev_action, prev_reward), device=self.device
        )
        if self.recurrent:
            model_inputs += (self.prev_rnn_state,)  # already on device
        model_outputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        action = self.distribution.sample(dist_info)
        agent_info = (dist_info,)
        if not self.actor_only:
            agent_info += (model_outputs.value,)
        if self.recurrent:
            rnn_state = model_outputs.next_rnn_state
            # on first step after reset, prev_rnn_state is None
            # must ensure prev_rnn_state matches expected data type and size
            prev_rnn_state = self.prev_rnn_state or buffer_func(
                rnn_state, torch.zeros_like)
            # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
            prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
            agent_info += (prev_rnn_state,)
            # overwrite self.prev_rnn_state with new rnn_state
            self.advance_rnn_state(rnn_state)  # keep on device

        agent_info = self.InfoCls(*agent_info)
        agent_step = AgentStep(action=action, agent_info=agent_info)
        return buffer_to(agent_step, device="cpu")

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to(
            (observation, prev_action, prev_reward), device=self.device
        )
        if self.recurrent:
            model_inputs += (self.prev_rnn_state,)  # already on device
        model_outputs = self.model(*model_inputs)
        value = model_outputs.value
        return buffer_to(value, device="cpu")
