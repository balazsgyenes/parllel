from abc import ABC, abstractmethod


class Callback(ABC):
    @abstractmethod
    def __call__(self, elapsed_steps: int) -> None:
        raise NotImplementedError
