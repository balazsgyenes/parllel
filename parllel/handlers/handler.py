from typing import Any, Optional, Union

import numpy as np

from parllel.handlers.agent import Agent, AgentStep
from parllel.buffers import Buffer


class Handler:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def step(self, observation: Buffer, previous_action: Buffer, previous_reward: Buffer, *,
        env_ids: Union[int, slice] = slice(None), out_action: Buffer = None, out_agent_info: Buffer = None,
    ) -> Optional[AgentStep]:
        agent_step: AgentStep = self._agent.step(
            np.asarray(observation), np.asarray(previous_action), np.asarray(previous_reward), env_ids=env_ids)
        if any(out is None for out in (out_action, out_agent_info)):
            return agent_step
        else:
            action, agent_info = agent_step
            out_action[:] = action
            out_agent_info[:] = agent_info

    def value(self, observation: Buffer, previous_action: Buffer, previous_reward: Buffer, *, 
        env_ids: Union[int, slice] = slice(None), out_value: Buffer = None,
    ) -> Optional[Buffer]:
        val: Buffer = self._agent.value(observation, previous_action, previous_reward, env_ids=env_ids)
        if out_value is None:
            return val
        else:
            out_value[:] = val

    def __getattr__(self, name: str) -> Any:
        if "_agent" in self.__dict__:
            return getattr(self._agent, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
