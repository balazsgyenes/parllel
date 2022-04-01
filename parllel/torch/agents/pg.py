from parllel.buffers import NamedArrayTupleClass


AgentInfo = NamedArrayTupleClass("PgAgentInfo", [
    "dist_info",
    "value",
    "prev_action",
])


AgentPrediction = NamedArrayTupleClass("PgAgentPrediction", [
    "dist_info",
    "value",
])
