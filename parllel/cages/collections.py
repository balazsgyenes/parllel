from dataclasses import dataclass
from typing import Dict, Tuple, Union

import gym
import numpy as np


ActionType = Union[np.ndarray, Dict[str, np.ndarray]]
ObsType = Union[np.ndarray, Dict[str, np.ndarray]]
RewardType = Union[float, np.ndarray, Dict[str, np.ndarray]]
DoneType = Union[bool, np.ndarray]
EnvInfoType = Dict

EnvStepType = Tuple[ObsType, RewardType, DoneType, EnvInfoType]
EnvRandomStepType = Tuple[ActionType, ObsType, RewardType, DoneType, EnvInfoType]


@dataclass
class EnvSpaces:
    observation: gym.Space
    action: gym.Space
