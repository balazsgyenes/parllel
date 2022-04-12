from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from parllel.arrays import Array
from parllel.buffers import Buffer

from .agent import Agent, AgentStep


class Handler(ABC):
    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    @abstractmethod
    def step(self, observation: Buffer[Array], *, env_indices: Union[int, slice] = ...,
            out_action: Buffer[Array] = None, out_agent_info: Buffer[Array] = None,
            ) -> Optional[AgentStep]:
        pass

    def value(self, observation: Buffer[Array], *, out_value: Buffer[Array] = None,
            ) -> Optional[Buffer]:
        raise NotImplementedError

    def initial_rnn_state(self, *, out_rnn_state: Buffer[Array] = None,
            )-> Buffer:
        raise NotImplementedError

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
