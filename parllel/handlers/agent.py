from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike

from parllel import Array, ArrayDict, ArrayLike, ArrayTree, Index


class Agent(ABC):
    @abstractmethod
    def step(
        self,
        observation: ArrayTree[Array],
        *,
        env_indices: Index = ...,
    ) -> tuple[ArrayTree, ArrayDict]:
        raise NotImplementedError

    def value(self, observation: ArrayTree[Array]) -> ArrayLike:
        raise NotImplementedError

    def initial_rnn_state(self) -> ArrayTree:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def reset_one(self, env_index: Index) -> None:
        pass

    def save_model(self, path: PathLike) -> None:
        pass

    def load_model(self, path: PathLike, *args, **kwargs) -> None:
        pass

    def train_mode(self, elapsed_steps: int) -> None:
        pass

    def sample_mode(self, elapsed_steps: int) -> None:
        pass

    def eval_mode(self, elapsed_steps: int) -> None:
        pass

    def close(self) -> None:
        pass
