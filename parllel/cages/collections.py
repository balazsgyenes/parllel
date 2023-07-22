from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat, Union

import gymnasium as gym
import numpy as np

from parllel import MappingTree

ActionType = MappingTree[np.ndarray]
ObsType = MappingTree[np.ndarray]
RewardType = Union[SupportsFloat, MappingTree[np.ndarray]]
DoneType = Union[bool, np.ndarray]
EnvInfoType = MappingTree

EnvStepType = tuple[ObsType, RewardType, DoneType, DoneType, DoneType, EnvInfoType]
EnvRandomStepType = tuple[
    ActionType,
    ObsType,
    RewardType,
    DoneType,
    DoneType,
    DoneType,
    EnvInfoType,
]
EnvResetType = tuple[ObsType, EnvInfoType]


@dataclass
class EnvSpaces:
    observation: gym.Space
    action: gym.Space
