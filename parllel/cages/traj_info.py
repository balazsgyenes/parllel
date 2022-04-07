from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict


@dataclass
class TrajInfo:
    Length: int = 0
    Return: float = 0
    NonzeroRewards: int = 0
    DiscountedReturn: float = 0
    _current_discount: float = 1.0
    _discount: float = 1.0

    def step(self, observation, action, reward, done, env_info):
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._current_discount * reward
        self._current_discount *= self._discount


_int_default_dict_factory = partial(defaultdict, int)
_float_default_dict_factory = partial(defaultdict, float)

@dataclass
class MultiAgentTrajInfo(TrajInfo):
    Return: Dict[str, float] = field(default_factory=_float_default_dict_factory)
    NonzeroRewards: Dict[str, int] = field(default_factory=_int_default_dict_factory)
    DiscountedReturn: Dict[str, float] = field(default_factory=_float_default_dict_factory)

    def step(self, observation, action, reward, done, env_info):
        self.Length += 1
        for agent_name, agent_reward in reward.items():
            self.Return[agent_name] += agent_reward
            self.NonzeroRewards[agent_name] += agent_reward != 0
            self.DiscountedReturn[agent_name] += self._current_discount * agent_reward
        self._current_discount *= self._discount
