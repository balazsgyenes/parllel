from dataclasses import dataclass
from typing import Union

from parllel.buffers import Buffer


@dataclass(frozen=True)
class AgentStep:
    action: Buffer
    agent_info: Buffer


class Agent:
    def initialize(self) -> None:
        raise NotImplementedError
    
    def step(self, observation: Buffer, previous_action: Buffer, previous_reward: Buffer, *,
        env_ids: Union[int, slice] = slice(None)
    ) -> AgentStep:
        raise NotImplementedError

    def value(self, observation: Buffer, previous_action: Buffer, previous_reward: Buffer, *,
        env_ids: Union[int, slice] = slice(None)
    ) -> Buffer:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def reset_one(self, env_index: int) -> None:
        raise NotImplementedError

    def sample_mode(self, elapsed_steps: int) -> None:
        pass

    def close(self) -> None:
        pass