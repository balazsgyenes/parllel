from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat, Union

import gymnasium as gym
import numpy as np

from parllel import ArrayOrMapping

ActionType = ArrayOrMapping[np.ndarray]
ObsType = ArrayOrMapping[np.ndarray]
RewardType = Union[SupportsFloat, ArrayOrMapping[np.ndarray]]
DoneType = Union[bool, np.ndarray]
EnvInfoType = ArrayOrMapping

EnvStepType = tuple[ObsType, RewardType, DoneType, DoneType, EnvInfoType]
EnvRandomStepType = tuple[
    ActionType,
    ObsType,
    RewardType,
    DoneType,
    DoneType,
    EnvInfoType,
]
EnvResetType = tuple[ObsType, EnvInfoType]


@dataclass
class EnvSpaces:
    observation: gym.Space
    action: gym.Space
