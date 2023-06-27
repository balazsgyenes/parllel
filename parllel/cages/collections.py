from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

ActionType = np.ndarray | Dict[str, np.ndarray]
ObsType = np.ndarray | Dict[str, np.ndarray]
RewardType = SupportsFloat | np.ndarray | Dict[str, np.ndarray]
TerminatedType = bool | np.ndarray
TruncatedType = bool | np.ndarray
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
