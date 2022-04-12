from parllel.buffers import NamedTupleClass

EnvStep = NamedTupleClass("EnvStep", 
    ["observation", "reward", "done", "env_info"])
EnvSpaces = NamedTupleClass("EnvSpaces",
    ["observation", "action"])
