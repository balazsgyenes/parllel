## Design Goals

rlpyt is a great piece of software, but there are several pain points when it comes to working with it.
- Mixing of code for RL and code for efficient utilization of hardware, whereas a typical workflow would involve getting an RL algorithm to function before attempting to accelerate it.
- Certain types have to be used together (e.g. GpuSampler and GpuCollectors)
- Complicated inheritance hierarchies with mixins, especially for agents.
- Duplicated code for cases that share a lot of similarity, especially for collectors, samplers.

## Design Decisions

- Add dependency injection to increase composability of different types. As much as possible, objects should not be responsible for initializing their dependent objects (e.g. sampler should not have to initialize collectors, agent, etc.). The initialization process is handled in the top-level script, assisted by patterns, functions that act as shortcuts for common use cases.
- The user should be able to easily control the contents of the sampler buffer. For this reason, the sampler should be kept as simple as possible to facilitate the writing of new sampler classes. Wherever possible, reading and writing data should be transparent (although this may be difficult if it also needs to be parallel).

## Design Issues

- The TrajInfo is currently handled by the cage, where agent info (e.g. value estimates, etc.) are not available. Is this an issue?
- How can we abstract how agents/models are shared across processes? Add a `models()` method to the agent for the handler to access all the models that need to be shared.
- The rlpyt sampling loop maintains local variables for the most recent, observation, action, reward, etc. These values are then copied one by one into the samples buffer. Is it more efficient to write directly into the samples buffer and provide views of it to read from?
- Can we support algorithms that are written for processing entire sequence (e.g. COMA), and therefore do not rely on bootstrap values? Can we support buffers with batch index as the leading dimension?
- Why can't a model be resetted mid-batch? Do we even need to consider wait-reset as an option? The sampler has full control of the rnn_state and can write zeroes/None if env is done
- Does it make sense to insist on NamedTuples and NamedArrayTuples everywhere, even when dicts might make more sense if the value needs to be modified. In any case, NamedArrayTuple/NamedTuple should have a `__repr__` that returns a dict for debug viewing.

## TODOs
- Implement (optional) out parameter for agent.step and cage.step methods. This reduces copying but also solves the problem of efficiently handling (parallel) write operations while keeping control in the sampler.
- Rename buffers to Arrays (or something else). Nomenclature:
    - Buffer is a (potentially nested) tuple/namedtuple/namedarraytuple of arrays
    - Arrays are either numpy ndarrays, (or a subclass) or torch tensors

## Ideas
- Sampler types:
    - ClassicSampler, which should cover most use cases
    - AlternatingSampler, which might provide better performance for slow environments
    - FeedForwardSampler, which is a simpler version that only works for non-recurrent models
    - FullEpisodeSampler, which returns only completed trajectories every iteration. This is essentially a configuration of the ClassicSampler (cages wait to reset, sampler stops if all envs done, samples buffer allocated with T equal to maximum episode length). Depending on wait-reset semantics, it might not make sense to have a separate class for this.
- Array types:
    - AlternatingArray wraps 2 arrays which alternate being written to (by the sampler) and read from (by the algorithm).
- NamedArrayTuples:
    - create metaclass to override behaviour of `type()`, so that getting the type of a Named\[Array\]Tuple returns the corresponding Named\[Array\]TupleClass. This enables code like `type(named_tup)(*iterable)`

