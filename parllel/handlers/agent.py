from abc import ABC, abstractmethod
from os import PathLike
from typing import Union

from parllel.buffers import Buffer, NamedTupleClass

AgentStep = NamedTupleClass("AgentStep", ["action", "agent_info"])


class Agent(ABC):
    @abstractmethod
    def step(self, observation: Buffer, *,
             env_indices: Union[int, slice] = ...) -> AgentStep:
        raise NotImplementedError

    def value(self, observation: Buffer) -> Buffer:
        raise NotImplementedError

    def initial_rnn_state(self) -> Buffer:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def reset_one(self, env_index: int) -> None:
        pass

    def save_model(self, path: PathLike) -> None:
        pass

    def train_mode(self, elapsed_steps: int) -> None:
        pass

    def sample_mode(self, elapsed_steps: int) -> None:
        pass

    def eval_mode(self, elapsed_steps: int) -> None:
        pass

    def close(self) -> None:
        pass
