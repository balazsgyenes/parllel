from dataclasses import dataclass
from typing import Optional, Union

import torch

from parllel.buffers import Buffer, buffer_map, buffer_method
from parllel.handlers.agent import AgentStep
from parllel.torch.distributions.categorical import Categorical, DistInfo
from parllel.torch.utils import buffer_to_device

from .agent import TorchAgent
from .pg import AgentInfo, AgentPrediction


@dataclass(frozen=True)
class ModelOutputs:
    pi: Buffer
    value: Buffer = None
    next_rnn_states: Buffer = None


class CategoricalPgAgent(TorchAgent):
    """Agent for policy gradient algorithm using categorical action
    distribution for discrete action spaces. Same as `GaussianPgAgent` except
    with `Categorical` distribution, and has a different interface to the model
    (model here outputs discrete probabilities in place of means and log_stds,
    while both output the value estimate).

    Assumptions of the model:
        - input arguments must include observation, may include previous_action,
            and may include rnn_states as an optional argument (in this order)
        - return type is the ModelOutputs dataclass defined in this module
    """
    def __init__(self, model: torch.nn.Module, distribution: Categorical,
                 device: torch.device = None,
                 ) -> None:
        super().__init__(model, distribution, device)
        self.recurrent = None

    @torch.no_grad()
    def dry_run(self, n_states: int, observation: Buffer,
            previous_action: Optional[Buffer] = None) -> AgentStep:
        
        model_inputs = (observation,)

        if previous_action is not None:
            previous_action = self._distribution.to_onehot(previous_action)
            model_inputs += (previous_action,)
        
        # model will generate an rnn_state even if we don't pass one
        model_inputs = buffer_to_device(model_inputs, device=self.device)

        model_outputs: ModelOutputs = self.model(*model_inputs)

        dist_info = DistInfo(prob=model_outputs.pi)
        action = self._distribution.sample(dist_info)

        # value may be None
        value = model_outputs.value

        # extend rnn_state by the number of environments the agent steps
        example_rnn_state = model_outputs.next_rnn_states
        if example_rnn_state is not None:
            self.recurrent = True

            # Extend an example_rnn_state to allocate enough space for each env.
            # repeat in batch dimension (shape should be [N,B,H])
            self._rnn_states = buffer_map(example_rnn_state,
                lambda t: torch.cat((t,) * n_states, dim=1))

            # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
            example_rnn_state = buffer_method(example_rnn_state, "transpose", 0, 1)

            # Stack previous action to allocate a slot for each env
            # Add a new leading dimension
            self._previous_action = buffer_map(previous_action,
                lambda t: torch.stack((t,) * n_states, dim=0))

            # TODO: verify leading dimensions of rnn_state, is the batch dimension empty?
            raise NotImplementedError
        else:
            self.recurrent = False

        agent_info = AgentInfo(dist_info=dist_info, value=value, prev_rnn_state=example_rnn_state)
        agent_step = AgentStep(action=action, agent_info=agent_info)
        return buffer_to_device(agent_step, device="cpu")

    @torch.no_grad()
    def step(self, observation: Buffer, *, env_indices: Union[int, slice] = ...,
             ) -> AgentStep:
        model_inputs = (observation,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        if self.recurrent:
            # already on device
            previous_action = self._previous_action[env_indices]
            previous_action = self._distribution.to_onehot(previous_action)
            rnn_states = self._rnn_states[env_indices]
            model_inputs += (previous_action, rnn_states)
        model_outputs: ModelOutputs = self.model(*model_inputs)

        # sample action from distribution returned by policy
        dist_info = DistInfo(prob=model_outputs.pi)
        action = self._distribution.sample(dist_info)

        # value may be None
        value = model_outputs.value

        # overwrite save rnn_state and action as inputs to next step
        prev_rnn_states = self.advance_states(model_outputs.next_rnn_states,
            action, env_indices)

        agent_info = AgentInfo(dist_info, value, prev_rnn_states)
        agent_step = AgentStep(action=action, agent_info=agent_info)
        return buffer_to_device(agent_step, device="cpu")

    @torch.no_grad()
    def value(self, observation: Buffer) -> Buffer:
        model_inputs = (observation,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        if self.recurrent:
            # already on device
            previous_action = self._distribution.to_onehot(self._previous_action)
            model_inputs += (previous_action, self._rnn_states)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        value = model_outputs.value
        return buffer_to_device(value, device="cpu")

    def predict(self, observation: Buffer, previous_action: Optional[Buffer] = None,
                init_rnn_states: Optional[Buffer] = None,
                ) -> AgentPrediction:
        """Performs forward pass on training data, for algorithm."""
        model_inputs = (observation,)
        if self.recurrent:
            # rnn_states were saved into the samples buffer as [B,N,H]
            # transform back [B,N,H] --> [N,B,H].
            previous_action = self._distribution.to_onehot(previous_action)
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            model_inputs += (previous_action, init_rnn_states,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        value = model_outputs.value
        prediction = AgentPrediction(dist_info, value)
        # prediction = buffer_to_device(prediction, device="cpu")
        return prediction
