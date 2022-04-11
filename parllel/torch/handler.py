from typing import Optional, Union

from nptyping import NDArray
from torch import Tensor

from parllel.arrays import Array
from parllel.buffers import Buffer
from parllel.handlers import Handler, AgentStep
from parllel.torch.agents.agent import TorchAgent
from parllel.torch.utils import numpify_buffer, torchify_buffer


class TorchHandler(Handler):
    def __init__(self, agent: TorchAgent) -> None:
        self._agent = agent

    def step(self, observation: Buffer[Array], *, env_indices:
            Union[int, slice] = ..., out_action: Buffer[Array] = None,
            out_agent_info: Buffer[Array] = None) -> Optional[AgentStep]:
        
        observation: Buffer[Tensor] = torchify_buffer(observation)

        agent_step: AgentStep = self._agent.step(observation,
            env_indices=env_indices)

        # torch tensors can be written directly into numpy arrays
        # agent_step = numpify_buffer(agent_step)

        if any(out is None for out in (out_action, out_agent_info)):
            return agent_step
        else:
            action, agent_info = agent_step
            out_action[:] = action
            out_agent_info[:] = agent_info

    def value(self, observation: Buffer[Array], *,
            out_value: Buffer[Array] = None) -> Optional[Buffer]:
        
        observation: Buffer[Tensor] = torchify_buffer(observation)

        value: Buffer[Tensor] = self._agent.value(observation)

        # torch tensors can be written directly into numpy arrays
        # value: Buffer[NDArray] = numpify_buffer(value)

        if out_value is None:
            return value
        else:
            out_value[:] = value

    def initial_rnn_state(self, *, out_rnn_state: Buffer[Array] = None,
            )-> Buffer:
        init_rnn_state: Buffer[Tensor] = self._agent.initial_rnn_state()

        # torch tensors can be written directly into numpy arrays
        # init_rnn_state: Buffer[NDArray] = numpify_buffer(init_rnn_state)

        if out_rnn_state is None:
            return init_rnn_state
        else:
            out_rnn_state[:] = init_rnn_state
