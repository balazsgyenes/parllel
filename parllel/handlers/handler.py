from typing import Any, Optional, Union

import numpy as np

from parllel.handlers.agent import Agent, AgentStep
from parllel.buffers import Buffer, buffer_func


class Handler:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def step(self, observation: Buffer, previous_action: Optional[Buffer], *,
             env_ids: Union[int, slice] = slice(None), out_action: Buffer = None, out_agent_info: Buffer = None,
             ) -> Optional[AgentStep]:
        """TODO: should the number of arguments be flexible here? It makes the code less readable.

        alternate implementation for this case:
        *(buffer_func(np.asarray, agent_input) for agent_input in agent_inputs),
        """

        agent_step: AgentStep = self._agent.step(
            buffer_func(np.asarray, observation),
            buffer_func(np.asarray, previous_action),
            env_ids=env_ids,
        )
        if any(out is None for out in (out_action, out_agent_info)):
            return agent_step
        else:
            action, agent_info = agent_step
            out_action[:] = action
            out_agent_info[:] = agent_info

    def value(self, observation: Buffer, previous_action: Optional[Buffer], *,
              env_ids: Union[int, slice] = slice(None), out_value: Buffer = None,
              ) -> Optional[Buffer]:
        val: Buffer = self._agent.value(
            buffer_func(np.asarray, observation),
            buffer_func(np.asarray, previous_action),
            env_ids=env_ids,
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
