from parllel.buffers import NamedArrayTupleClass


AgentInfo = NamedArrayTupleClass("PgAgentInfo", [
    "dist_info",
    "value",
    "prev_rnn_state",
])


AgentPrediction = NamedArrayTupleClass("PgAgentPrediction", [
    "dist_info",
    "value",
    "next_rnn_state",
])
