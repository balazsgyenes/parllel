from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np

ActionType = Union[np.ndarray, Dict[str, np.ndarray]]
ObsType = Union[np.ndarray, Dict[str, np.ndarray]]
RewardType = Union[SupportsFloat, np.ndarray, Dict[str, np.ndarray]]
TerminatedType = Union[bool, np.ndarray]
TruncatedType = Union[bool, np.ndarray]
EnvInfoType = Dict

EnvStepType = Tuple[ObsType, RewardType, TerminatedType, TruncatedType, EnvInfoType]
EnvRandomStepType = Tuple[
    ActionType, ObsType, RewardType, TerminatedType, TruncatedType, EnvInfoType
]
EnvResetType = Tuple[ObsType, EnvInfoType]


@dataclass
class EnvSpaces:
    observation: gym.Space
    action: gym.Space
