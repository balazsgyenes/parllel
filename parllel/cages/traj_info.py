from collections import defaultdict
from dataclasses import astuple, dataclass, field, fields
from functools import partial
from typing import Any, ClassVar, Dict, Iterator, Tuple, Union

import numpy as np


@dataclass
class TrajInfo:
    _discount: ClassVar[float] = 1.0  # class variable shared by all instances
    _current_discount: float = 1.0
    Length: int = 0
    Return: float = 0
    NonzeroRewards: int = 0
    DiscountedReturn: float = 0

    @classmethod
    def set_discount(cls, discount: float) -> None:
        cls._discount = discount

    def step(self,
        observation: Any,
        action: Any,
        reward: Union[np.array, float],
        done: bool,
        env_info: Dict[str, Any],
    ) -> None:
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._current_discount * reward
        self._current_discount *= self._discount


@dataclass
class MultiAgentTrajInfo(TrajInfo):
    Return: Dict[str, float] = field(default_factory=partial(defaultdict, float))
    NonzeroRewards: Dict[str, int] = field(default_factory=partial(defaultdict, int))
    DiscountedReturn: Dict[str, float] = field(default_factory=partial(defaultdict, float))

    def step(self,
        observation: Any,
        action: Any,
        reward: Dict[str, Union[np.array, float]],
        done: bool,
        env_info: Dict[str, Any],
    ) -> None:
        self.Length += 1
        for agent_name, agent_reward in reward.items():
            self.Return[agent_name] += agent_reward
            self.NonzeroRewards[agent_name] += agent_reward != 0
            self.DiscountedReturn[agent_name] += self._current_discount * agent_reward
        self._current_discount *= self._discount


def zip_trajectories(*trajs: Tuple[TrajInfo]) -> Iterator[Tuple[Any, ...]]:
    """A generator that yields key, value1, value2, value3, ... for all
    dataclass objects passed as arguments. All objects must be instances of
    the same dataclass.
    """
    if not trajs:
        return

    # Since they are all the same type, all TrajInfos will have the same fields
    # in the same order.
    field_names = (field.name for field in fields(trajs[0]))

    # All tuples guaranteed to have corresponding values in the same order.
    yield from zip(field_names, *(astuple(traj) for traj in trajs))
