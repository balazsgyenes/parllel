from dataclasses import dataclass

@dataclass
class TrajInfo:
    Length: int
    Return: float
    NonZeroRewards: int
    DiscountedReturn: float
    _current_discount: float = 1.0
    _discount: float = 1.0

    def step(self, observation, reward, done, env_info):
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._current_discount * reward
        self._current_discount *= self._discount
