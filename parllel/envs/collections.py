from parllel.types.named_tuple import NamedTupleType

EnvStep = NamedTupleType("EnvStep", 
    ["observation", "reward", "done", "env_info"])
