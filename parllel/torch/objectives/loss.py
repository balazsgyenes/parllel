from torch import Tensor

from parllel import ArrayDict
from parllel.algorithm import AlgoInfoType


class Loss:
    def loss(self, samples: ArrayDict[Tensor]) -> tuple[Tensor, AlgoInfoType]:
        raise NotImplementedError
