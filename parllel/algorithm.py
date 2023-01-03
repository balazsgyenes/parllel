from abc import ABC, abstractmethod
from typing import Dict, List, Union

from parllel.buffers import Samples


class Algorithm(ABC):
    @abstractmethod
    def optimize_agent(self, elapsed_steps: int, samples: Samples) -> Dict[str, Union[int, List[float]]]:
        pass
