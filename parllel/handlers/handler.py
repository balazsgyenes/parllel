from typing import Any, Optional, Union

import numpy as np

from parllel.handlers.agent import Agent, AgentStep
from parllel.buffers import Buffer, buffer_map


class Handler:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def dry_run(self, n_states: int, observation: Buffer, previous_action: Optional[Buffer] = None,
                ) -> AgentStep:
        observation, previous_action = buffer_map(np.asarray,(observation, previous_action))

        example = self._agent.dry_run(n_states, observation, previous_action)

        return example

    def step(self, observation: Buffer, previous_action: Optional[Buffer] = None,
             *, env_indices: Union[int, slice] = ..., out_action: Buffer = None,
             out_agent_info: Buffer = None,
             ) -> Optional[AgentStep]:

        observation, previous_action = buffer_map(np.asarray,(observation, previous_action))

        agent_step: AgentStep = self._agent.step(observation, previous_action,
                                                 env_indices=env_indices)

        if any(out is None for out in (out_action, out_agent_info)):
            return agent_step
        else:
            action, agent_info = agent_step
            out_action[:] = action
            out_agent_info[:] = agent_info

    def value(self, observation: Buffer, previous_action: Optional[Buffer] = None,
              *, out_value: Buffer = None,
              ) -> Optional[Buffer]:
        observation, previous_action = buffer_map(np.asarray,(observation, previous_action))

        value: Buffer = self._agent.value(observation, previous_action)

        if out_value is None:
            return value
        else:
            out_value[:] = value

    def __getattr__(self, name: str) -> Any:
        if "_agent" in self.__dict__:
            return getattr(self._agent, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
