from parllel.types.named_tuple import NamedArrayTupleType

Samples = NamedArrayTupleType("Samples", ["agent", "env"])
AgentSamples = NamedArrayTupleType("AgentSamples",
    ["action", "prev_action", "agent_info"])
EnvSamples = NamedArrayTupleType("EnvSamples",
    ["observation", "reward", "prev_reward", "done", "env_info"])