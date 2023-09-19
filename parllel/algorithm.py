from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

from parllel import Array, ArrayDict

AlgoInfoType = dict[str, Union[int, float, list[int], list[float]]]


class Algorithm(ABC):
    @abstractmethod
    def optimize_agent(
        self,
        elapsed_steps: int,
        samples: ArrayDict[Array],
    ) -> AlgoInfoType:
        pass
