from abc import ABC, abstractmethod
from typing import Optional, Union

from parllel.buffers import Buffer, NamedTupleClass

AgentStep = NamedTupleClass("AgentStep", ["action", "agent_info"])


class Agent(ABC):
    @abstractmethod
    def step(self, observation: Buffer, previous_action: Optional[Buffer] = None,
             *, env_indices: Union[int, slice] = ...,
             ) -> AgentStep:
        raise NotImplementedError

    def value(self, observation: Buffer, previous_action: Optional[Buffer] = None,
             ) -> Buffer:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def reset_one(self, env_index: int) -> None:
        pass

    def sample_mode(self, elapsed_steps: int) -> None:
        pass

    def close(self) -> None:
        pass
