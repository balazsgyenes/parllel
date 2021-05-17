from dataclasses import dataclass
from typing import Any

@dataclass
class AgentInfo:
    dist_info: Any
    value: Any
    prev_rnn_state: Any = None


# namedarraytuples defining the structure of agent_info returned by `step`
AgentInfo = NamedArrayTupleSchema("AgentInfoRnn",
    ["dist_info", "value", "prev_rnn_state"])
ActorInfo = NamedArrayTupleSchema("ActorInfoRnn", ["dist_info", "prev_rnn_state"])

# namedarraytuples defining the structure returned by `__call__`
AgentOutputs = NamedArrayTupleSchema("AgentOutputs", ["dist_info", "value", "next_rnn_state"])
ActorOutputs = NamedArrayTupleSchema("ActorOutputs", ["dist_info", "next_rnn_state"])

