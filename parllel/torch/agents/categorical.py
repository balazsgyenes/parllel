from dataclasses import dataclass
from typing import Optional, Tuple, Union

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
    next_rnn_state: Buffer = None


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
            example_action: Optional[Buffer] = None) -> Tuple[AgentInfo, Buffer]:
        
        model_inputs = (observation,)

        if example_action is not None:
            prev_act_onehot = self._distribution.to_onehot(example_action)
            model_inputs += (prev_act_onehot,)
        
        # model will generate an rnn_state even if we don't pass one
        model_inputs = buffer_to_device(model_inputs, device=self.device)

        model_outputs: ModelOutputs = self.model(*model_inputs)

        dist_info = DistInfo(prob=model_outputs.pi)

        # value may be None
        value = model_outputs.value

        # extend rnn_state by the number of environments the agent steps
        rnn_state = model_outputs.next_rnn_state
        if rnn_state is not None:
            self.recurrent = True

            # Extend an rnn_state to allocate enough space for each env.
            # repeat in batch dimension (shape should be [N,B,H])
            self._rnn_states = buffer_map(
                lambda t: torch.cat((t,) * n_states, dim=1),
                rnn_state,
            )

            # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
            rnn_state = buffer_method(rnn_state, "transpose", 0, 1)
            # remove batch dimension
            rnn_state = rnn_state[0]

            # Stack previous action to allocate a slot for each env
            # Add a new leading dimension
            # if None, this has no effect
            previous_action = buffer_map(
                lambda t: torch.stack((t,) * n_states, dim=0),
                example_action,
            )
            self._previous_action = buffer_to_device(previous_action,
                device=self.device)
        else:
            self.recurrent = False

        agent_info = AgentInfo(dist_info=dist_info, value=value,
            prev_action=example_action)
        return buffer_to_device((agent_info, rnn_state), device="cpu")

    @torch.no_grad()
    def step(self, observation: Buffer, *, env_indices: Union[int, slice] = ...,
             ) -> AgentStep:
        model_inputs = (observation,)
        model_inputs = buffer_to_device(model_inputs, device=self.device)
        if self.recurrent:
            # already on device
            rnn_states, previous_action = self._get_states(env_indices)
            previous_action = self._distribution.to_onehot(previous_action)
            model_inputs += (previous_action, rnn_states)
        model_outputs: ModelOutputs = self.model(*model_inputs)

        # sample action from distribution returned by policy
        dist_info = DistInfo(prob=model_outputs.pi)
        action = self._distribution.sample(dist_info)

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
            previous_action = self._distribution.to_onehot(self._previous_action)
            model_inputs += (previous_action, rnn_states)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        value = model_outputs.value
        return buffer_to_device(value, device="cpu")

    def predict(self, observation: Buffer, agent_info: AgentInfo,
                init_rnn_state: Optional[Buffer] = None,
                ) -> AgentPrediction:
        """Performs forward pass on training data, for algorithm."""
        model_inputs = (observation,)
        if self.recurrent:
            # rnn_states were saved into the samples buffer as [B,N,H]
            # transform back [B,N,H] --> [N,B,H].
            previous_action = agent_info.prev_action
            previous_action = self._distribution.to_onehot(previous_action)
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            model_inputs += (previous_action, init_rnn_state,)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        value = model_outputs.value
        prediction = AgentPrediction(dist_info, value)
        return prediction
