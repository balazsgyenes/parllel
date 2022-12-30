from parllel.buffers import NamedTupleClass

# TODO: replace with dataclasses
EnvStep = NamedTupleClass("EnvStep", 
    ["observation", "reward", "done", "env_info"])
EnvSpaces = NamedTupleClass("EnvSpaces",
    ["observation", "action"])
