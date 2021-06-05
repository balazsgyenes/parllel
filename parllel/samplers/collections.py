from parllel.buffers import NamedArrayTupleType

Samples = NamedArrayTupleType("Samples", ["agent", "env"])
AgentSamples = NamedArrayTupleType("AgentSamples",
    ["action", "agent_info"])
EnvSamples = NamedArrayTupleType("EnvSamples",
    ["observation", "reward", "done", "env_info"])