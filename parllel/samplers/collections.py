from parllel.buffers import NamedArrayTupleClass

Samples = NamedArrayTupleClass("Samples", ["agent", "env"])
AgentSamples = NamedArrayTupleClass("AgentSamples",
    ["action", "agent_info"])
EnvSamples = NamedArrayTupleClass("EnvSamples",
    ["observation", "reward", "done", "env_info"])