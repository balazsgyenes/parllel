from parllel.buffers import NamedArrayTupleClass

Samples = NamedArrayTupleClass("Samples", ["agent", "env"])
AgentSamples = NamedArrayTupleClass("AgentSamples",
    ["action", "agent_info"])
AgentSamplesWBootstrap = NamedArrayTupleClass("AgentSamplesWithBootstrapValue",
    ["action", "agent_info", "bootstrap_value"])
EnvSamples = NamedArrayTupleClass("EnvSamples",
    ["observation", "reward", "done", "env_info"])