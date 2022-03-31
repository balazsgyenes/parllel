from typing import Any, Optional, Union

import numpy as np
from nptyping import NDArray

from parllel.arrays import Array
from parllel.buffers import Buffer, buffer_map

from .agent import Agent, AgentStep


class Handler:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def step(self, observation: Buffer[Array], *, env_indices: Union[int, slice] = ...,
            out_action: Buffer[Array] = None, out_agent_info: Buffer[Array] = None,
            ) -> Optional[AgentStep]:

        observation: Buffer[NDArray] = buffer_map(np.asarray, observation)

        agent_step: AgentStep = self._agent.step(observation,
                                                 env_indices=env_indices)

        if any(out is None for out in (out_action, out_agent_info)):
            return agent_step
        else:
            action, agent_info = agent_step
            out_action[:] = action
            out_agent_info[:] = agent_info

    def value(self, observation: Buffer[Array], *, out_value: Buffer[Array] = None,
            ) -> Optional[Buffer]:
        observation = buffer_map(np.asarray, observation)

        value: Buffer[Array] = self._agent.value(observation)

        if out_value is None:
            return value
        else:
            out_value[:] = value

    def reset(self) -> None:
        self._agent.reset()

    def reset_one(self, env_index: int) -> None:
        self._agent.reset_one(env_index)

    def sample_mode(self, elapsed_steps: int) -> None:
        self._agent.sample_mode(elapsed_steps)

    def close(self) -> None:
        self._agent.close()

    def __getattr__(self, name: str) -> Any:
        if "_agent" in self.__dict__:
            return getattr(self._agent, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
