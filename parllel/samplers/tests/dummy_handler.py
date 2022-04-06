from typing import Optional, Union

from nptyping import NDArray

from parllel.arrays import Array
from parllel.buffers import Buffer, buffer_asarray
from parllel.handlers import AgentStep, Handler


class DummyHandler(Handler):
    def step(self, observation: Buffer[Array], *, env_indices:
            Union[int, slice] = ..., out_action: Buffer[Array] = None,
            out_agent_info: Buffer[Array] = None) -> Optional[AgentStep]:
        
        observation: Buffer[NDArray] = buffer_asarray(observation)

        agent_step: AgentStep = self._agent.step(observation,
            env_indices=env_indices)

        if any(out is None for out in (out_action, out_agent_info)):
            return agent_step
        else:
            action, agent_info = agent_step
            out_action[:] = action
            out_agent_info[:] = agent_info

    def value(self, observation: Buffer[Array], *,
            out_value: Buffer[Array] = None) -> Optional[Buffer]:
        
        observation: Buffer[NDArray] = buffer_asarray(observation)

        value: Buffer[NDArray] = self._agent.value(observation)

        if out_value is None:
            return value
        else:
            out_value[:] = value
