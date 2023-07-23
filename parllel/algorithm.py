from __future__ import annotations

from abc import ABC, abstractmethod

from parllel import Array, ArrayDict


class Algorithm(ABC):
    @abstractmethod
    def optimize_agent(
        self,
        elapsed_steps: int,
        samples: ArrayDict[Array],
    ) -> dict[str, int | list[float]]:
        pass
