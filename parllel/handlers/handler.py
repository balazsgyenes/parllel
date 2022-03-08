from typing import Any, Optional, Union

import numpy as np

from parllel.handlers.agent import Agent, AgentStep
from parllel.buffers import Buffer, buffer_func


class Handler:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def step(self, observation: Buffer, previous_action: Optional[Buffer] = None, *,
             env_indices: Union[int, slice] = ..., out_action: Buffer = None, out_agent_info: Buffer = None,
             ) -> Optional[AgentStep]:

        observation, previous_action = buffer_func(np.asarray((observation, previous_action)))

        agent_step: AgentStep = self._agent.step(observation, previous_action, env_indices)

        if any(out is None for out in (out_action, out_agent_info)):
            return agent_step
        else:
            action, agent_info = agent_step
            out_action[:] = action
            out_agent_info[:] = agent_info

    def value(self, observation: Buffer, previous_action: Optional[Buffer], *,
              env_indices: Union[int, slice] = ..., out_value: Buffer = None,
              ) -> Optional[Buffer]:
        val: Buffer = self._agent.value(
            buffer_func(np.asarray, observation),
            buffer_func(np.asarray, previous_action),
            env_indices=env_indices,
        )
        if out_value is None:
            return val
        else:
            out_value[:] = val

    def __getattr__(self, name: str) -> Any:
        if "_agent" in self.__dict__:
            return getattr(self._agent, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
