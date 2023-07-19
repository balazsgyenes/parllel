from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat, Union

import gymnasium as gym
import numpy as np

ActionType = Union[np.ndarray, dict[str, np.ndarray]]
ObsType = Union[np.ndarray, dict[str, np.ndarray]]
RewardType = Union[SupportsFloat, np.ndarray, dict[str, np.ndarray]]
DoneType = Union[bool, np.ndarray]
EnvInfoType = dict

EnvStepType = tuple[ObsType, RewardType, DoneType, DoneType, DoneType, EnvInfoType]
EnvRandomStepType = tuple[
    ActionType, ObsType, RewardType, DoneType, DoneType, DoneType, EnvInfoType,
]
EnvResetType = tuple[ObsType, EnvInfoType]


@dataclass
class EnvSpaces:
    observation: gym.Space
    action: gym.Space
