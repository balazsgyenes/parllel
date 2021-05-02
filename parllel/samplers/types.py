from parllel.types.named_tuple import NamedArrayTupleType

Samples = NamedArrayTupleType("Samples", ["agent", "env"])
AgentSamples = NamedArrayTupleType("AgentSamples",
    ["action", "agent_info", "rnn_state"])
EnvSamples = NamedArrayTupleType("EnvSamples",
    ["observation", "reward", "done", "env_info"])