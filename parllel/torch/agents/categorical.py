from dataclasses import dataclass
from typing import Optional, Union

import torch

from parllel.buffers import Buffer, buffer_method, buffer_asarray
from parllel.handlers.agent import AgentStep
from parllel.torch.distributions.categorical import Categorical, DistInfo
from parllel.torch.utils import buffer_to_device, torchify_buffer

from .agent import TorchAgent
from .pg import AgentInfo, AgentPrediction


@dataclass(frozen=True)
class ModelOutputs:
    pi: Buffer
    value: Optional[Buffer] = None
    next_rnn_state: Optional[Buffer] = None


class CategoricalPgAgent(TorchAgent):
    """Agent for policy gradient algorithm using categorical action
    distribution for discrete action spaces.
    
    The model must return the ModelOutputs type in this module, which contains:
        - pi: probabilities for each discrete action in the action space
        - value: value estimates, which can be omitted in cases without value
            prediction (e.g. vanilla PG) or where another entity predicts
            value (multi-agent scenarious)
        - next_rnn_state: the hidden recurrent state for the next time step

    The model must take between 1-3 arguments in the following order (arguments
    are only positional, not passed by keyword):
        - observation: current state of the environment
        - previous_action: action sampled from distribution from last time step
        - rnn_state: hidden recurrent state from last time step
    """
    def __init__(self,
            model: torch.nn.Module,
            distribution: Categorical,
            example_obs: Buffer,
            example_action: Optional[Buffer] = None,
            device: Optional[torch.device] = None,
            recurrent: bool = False,
        ) -> None:
        super().__init__(model, distribution, device)

        self.recurrent = recurrent

        example_obs = buffer_asarray(example_obs)
        example_obs = torchify_buffer(example_obs)
        example_inputs = (example_obs,)
        
        if self.recurrent:
            if example_action is None:
                raise ValueError(
                    "An example of an action is required for recurrent models."
                )
            example_action = buffer_asarray(example_action)
            example_action = torchify_buffer(example_action)
            example_act_onehot = self.distribution.to_onehot(example_action)
            example_inputs += (example_act_onehot,)

        # model will generate an rnn_state even if we don't pass one
        example_inputs = buffer_to_device(example_inputs, device=self.device)

        with torch.no_grad():
            try:
                model_outputs: ModelOutputs = self.model(*example_inputs)
            except TypeError as e:
                raise TypeError("You may have forgotten to pass recurrent=True"
                    " when creating this agent.") from e

        if self.recurrent:
            # store persistent agent state on device for next step
            self.rnn_states = model_outputs.next_rnn_state
            self.previous_action = buffer_to_device(example_action,
                device=self.device)

    @torch.no_grad()
    def step(self,
            observation: Buffer,
            *,
            env_indices: Union[int, slice] = ...,
        ) -> AgentStep:
        model_inputs = (observation,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        if self.recurrent:
            # already on device
            rnn_states, previous_action = self._get_states(env_indices)
            previous_action = self.distribution.to_onehot(previous_action)
            model_inputs += (previous_action, rnn_states)
        model_outputs: ModelOutputs = self.model(*model_inputs)

        # sample action from distribution returned by policy
        dist_info = DistInfo(prob=model_outputs.pi)
        action = self.distribution.sample(dist_info)

        # value may be None
        value = model_outputs.value

        if self.recurrent:
            # overwrite saved rnn_state and action as inputs to next step
            previous_action = self._advance_states(
                model_outputs.next_rnn_state, action, env_indices)
        else:
            previous_action = None

        agent_info = AgentInfo(dist_info, value, previous_action)
        agent_step = AgentStep(action=action, agent_info=agent_info)
        return buffer_to_device(agent_step, device="cpu")

    @torch.no_grad()
    def initial_rnn_state(self) -> Buffer:
        # transpose the rnn_states from [N,B,H] -> [B,N,H] for storage.
        init_rnn_state, _ = self._get_states(...)
        init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
        return buffer_to_device(init_rnn_state, device="cpu")

    @torch.no_grad()
    def value(self, observation: Buffer) -> Buffer:
        model_inputs = (observation,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        if self.recurrent:
            # already on device
            rnn_states, previous_action = self._get_states(...)          
            previous_action = self.distribution.to_onehot(self.previous_action)
            model_inputs += (previous_action, rnn_states)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        value = model_outputs.value
        return buffer_to_device(value, device="cpu")

    def predict(self,
            observation: Buffer,
            agent_info: AgentInfo,
            init_rnn_state: Optional[Buffer] = None,
        ) -> AgentPrediction:
        """Performs forward pass on training data, for algorithm."""
        model_inputs = (observation,)
        if self.recurrent:
            # rnn_states were saved into the samples buffer as [B,N,H]
            # transform back [B,N,H] --> [N,B,H].
            previous_action = agent_info.prev_action
            previous_action = self.distribution.to_onehot(previous_action)
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            model_inputs += (previous_action, init_rnn_state,)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        value = model_outputs.value
        prediction = AgentPrediction(dist_info, value)
        return prediction
