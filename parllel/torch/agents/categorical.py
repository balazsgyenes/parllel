from dataclasses import dataclass
from typing import Optional, Union

import torch

from parllel.handlers.agent import AgentStep
from parllel.torch.distributions.categorical import Categorical, DistInfo
from parllel.buffers import Buffer, buffer_func, buffer_method
from parllel.torch.utils import buffer_to_device

from .agent import TorchAgent
from .pg import AgentInfo, AgentPrediction


@dataclass(frozen=True)
class ModelOutputs:
    pi: Buffer
    value: Buffer = None
    next_rnn_states: Buffer = None


class CategoricalPgAgent(TorchAgent):
    """
    Agent for policy gradient algorithm using categorical action distribution.
    Same as ``GausssianPgAgent`` and related classes, except uses
    ``Categorical`` distribution, and has a different interface to the model
    (model here outputs discrete probabilities in place of means and log_stds,
    while both output the value estimate).

    Assumptions on the model:
        - input arguments must include observation, may include previous_action,
            and may include rnn_states as an optional argument (in this order)
        - return type is the ModelOutputs dataclass defined in this module
    """
    def __init__(self, model: torch.nn.Module, distribution: Categorical, device: torch.device = None) -> None:
        super().__init__(model, distribution, device)
        self.recurrent = False

    @torch.no_grad()
    def dry_run(self, n_states: int, observation: Buffer, previous_action: Optional[Buffer] = None) -> Buffer:
        model_inputs = (observation,)

        if previous_action is not None:
            previous_action = self._distribution.to_onehot(previous_action)
            model_inputs += (previous_action,)

        model_inputs = buffer_to_device(model_inputs, device=self.device)

        model_outputs: ModelOutputs = self.model(*model_inputs)

        dist_info = DistInfo(prob=model_outputs.pi)
        action = self.distribution.sample(dist_info)

        # value may be None
        value = model_outputs.value

        # extend rnn_state by the number of environments the agent steps
        example_rnn_state = model_outputs.next_rnn_states
        if example_rnn_state is not None:
            self.recurrent = True

            def extend_rnn_state(example_rnn_state):
                """Extend an example_rnn_state to allocate enough space for
                each env.
                """
                # duplicate as many times as requested
                rnn_states = (example_rnn_state) * n_states
                # concatenate in B dimension (shape should be [N,B,H])
                return torch.cat(rnn_states, dim=1)

            self._rnn_states = buffer_func(example_rnn_state, extend_rnn_state)

            # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
            example_rnn_state = buffer_method(example_rnn_state, "transpose", 0, 1)

            # TODO: verify leading dimensions of rnn_state, is the batch dimension empty?
            raise NotImplementedError

        agent_info = AgentInfo(dist_info=dist_info, value=value, prev_rnn_state=example_rnn_state)
        agent_step = AgentStep(action=action, agent_info=agent_info)
        return buffer_to_device(agent_step, device="cpu")

    @torch.no_grad()
    def step(self, observation: Buffer, previous_action: Buffer = None, env_indices: Union[int, slice] = ...):
        model_inputs = (observation,)
        if previous_action is not None:
            previous_action = self._distribution.to_onehot(previous_action)
            model_inputs += (previous_action,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        if self.recurrent:
            model_inputs += (self._rnn_states,)  # already on device
        model_outputs: ModelOutputs = self.model(*model_inputs)

        # sample action from distribution returned by policy
        dist_info = DistInfo(prob=model_outputs.pi)
        action = self.distribution.sample(dist_info)

        # value may be None
        value = model_outputs.value

        # save previous rnn states to samples buffer
        # transpose the rnn_states from [N,B,H] --> [B,N,H] for storage.
        prev_rnn_states = buffer_method(self._rnn_states, "transpose", 0, 1)

        # overwrite self._rnn_states with new rnn_states (although it may be None)
        # keep on device
        self.advance_rnn_states(model_outputs.next_rnn_states, env_indices)

        agent_info = AgentInfo(dist_info, value, prev_rnn_states)
        agent_step = AgentStep(action=action, agent_info=agent_info)
        return buffer_to_device(agent_step, device="cpu")

    @torch.no_grad()
    def value(self, observation: Buffer, previous_action: Buffer = None):
        model_inputs = (observation,)
        if previous_action is not None:
            previous_action = self._distribution.to_onehot(previous_action)
            model_inputs += (previous_action,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        if self.recurrent:
            model_inputs += (self._rnn_states,)  # already on device
        model_outputs: ModelOutputs = self.model(*model_inputs)
        value = model_outputs.value
        return buffer_to_device(value, device="cpu")

    def predict(self, observation: Buffer, previous_action: Buffer = None, init_rnn_states: Buffer = None):
        """Performs forward pass on training data, for algorithm."""
        model_inputs = (observation,)
        if previous_action is not None:
            previous_action = self._distribution.to_onehot(previous_action)
            model_inputs += (previous_action,)
        if self.recurrent:
            model_inputs += (init_rnn_states,)  # already on device
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        value = model_outputs.value
        output = buffer_to_device((dist_info, value), device="cpu")
        return AgentPrediction(*output)
