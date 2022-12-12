from dataclasses import dataclass
from typing import Optional, Union

import gym
import torch

from parllel.buffers import Buffer, buffer_map, buffer_method, dict_to_namedtuple
from parllel.handlers.agent import AgentStep
from parllel.torch.distributions.gaussian import Gaussian, DistInfoStd
from parllel.torch.utils import buffer_to_device, torchify_buffer

from .agent import TorchAgent
from .pg import AgentInfo, AgentPrediction


@dataclass(frozen=True)
class ModelOutputs:
    mean: Buffer
    log_std: Buffer
    value: Optional[Buffer] = None
    next_rnn_state: Optional[Buffer] = None


class GaussianPgAgent(TorchAgent):
    """Agent for policy gradient algorithm using gaussian action
    distribution for continuous action spaces. 
    TODO

    Assumptions of the model:
        - input arguments must include observation, may include previous_action,
            and may include rnn_states as an optional argument (in this order)
        - return type is the ModelOutputs dataclass defined in this module
    """
    def __init__(self,
            model: torch.nn.Module,
            distribution: Gaussian,
            observation_space: gym.Space,
            action_space: gym.Space,
            n_states: Optional[int] = None,
            device: Optional[torch.device] = None,
            recurrent: bool = False,
        ) -> None:
        super().__init__(model, distribution, device)

        self.observation_space = observation_space
        self.action_space = action_space
        self.recurrent = recurrent

        if recurrent and n_states is None:
            raise ValueError(
                "For a recurrent model, please pass the number of recurrent "
                "states this agent should store (this is usually the number "
                "of environments)."
            )

        example_obs = self.observation_space.sample()
        example_obs = dict_to_namedtuple(example_obs, "observation")
        example_obs = torchify_buffer(example_obs)
        example_inputs = (example_obs,)
        
        if self.recurrent:
            example_action = self.action_space.sample()
            example_action = dict_to_namedtuple(example_action, "action")
            example_action = torchify_buffer(example_action)
            example_inputs += (example_action,)

        # model will generate an rnn_state even if we don't pass one
        example_inputs = buffer_to_device(example_inputs, device=self.device)

        with torch.no_grad():
            try:
                model_outputs: ModelOutputs = self.model(*example_inputs)
            except TypeError as e:
                # TODO e is not actually printed.
                raise TypeError("You may have forgotten to pass recurrent=True"
                    " when creating this agent.") from e

        if self.recurrent:
            # Extend an rnn_state to allocate a slot for each env.
            # repeat in batch dimension (shape should be [N,B,H])
            rnn_state = model_outputs.next_rnn_state
            self.rnn_states = buffer_map(
                lambda t: torch.cat((t,) * n_states, dim=1),
                rnn_state,
            )

            # Stack previous action to allocate a slot for each env
            # Add a new leading dimension
            previous_action = buffer_map(
                lambda t: torch.stack((t,) * n_states, dim=0),
                example_action,
            )
            self.previous_action = buffer_to_device(previous_action,
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
            model_inputs += (previous_action, rnn_states)
        model_outputs: ModelOutputs = self.model(*model_inputs)

        # sample action from distribution returned by policy
        dist_info = DistInfoStd(
            mean=model_outputs.mean,
            log_std=model_outputs.log_std,
        )
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
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            model_inputs += (previous_action, init_rnn_state,)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        dist_info = DistInfoStd(
            mean=model_outputs.mean,
            log_std=model_outputs.log_std,
        )
        value = model_outputs.value
        prediction = AgentPrediction(dist_info, value)
        return prediction
