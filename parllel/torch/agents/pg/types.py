from dataclasses import dataclass
from typing import Any

# TODO: should these be NamedTuples?

@dataclass(frozen=True)
class AgentInfo:
    dist_info: Any
    value: Any = None
    prev_rnn_state: Any = None


@dataclass(frozen=True)
class AgentEvaluation:
    dist_info: Any
    value: Any = None
    next_rnn_state: Any = None
