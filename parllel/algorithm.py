from abc import ABC, abstractmethod

from parllel.buffers import Samples


class Algorithm(ABC):
    @abstractmethod
    def optimize_agent(self, elapsed_steps: int, samples: Samples):
        pass
