## Design Goals

rlpyt is a great piece of software, but there are several pain points when it comes to working with it.
- Mixing of code for RL and code for efficient utilization of hardware, whereas a typical workflow would involve getting an RL algorithm to function before attempting to accelerate it.
- Certain types have to be used together (e.g. GpuSampler and GpuCollectors)
- Complicated inheritance hierarchies with mixins, especially for agents.
- Duplicated code for cases that share a lot of similarity, especially for collectors, samplers.

## Design Decisions

- Add dependency injection to increase composability of different types. As much as possible, objects should not be responsible for initializing their dependent objects (e.g. sampler should not have to initialize collectors, agent, etc.). The initialization process is handled in the top-level script, assisted by patterns, functions that act as shortcuts for common use cases.
- The user should be able to easily control the contents of the sampler buffer. For this reason, the sampler should be kept as simple as possible to facilitate the writing of new sampler classes. Wherever possible, reading and writing data should be transparent (although this may be difficult if it also needs to be parallel).
- Initialization takes as input an example of the input the object will see during the training loop, and return an example of its output. This allows all objects to pre-allocate memory and data types for efficiency, without hard-coding dependencies.
- Code for RL and code for optimization should remain separated. Instead of combining these objects through inheritance, one should wrap the other (e.g. Handler wraps Agent instead of handler code being in base class of Agent).
- Sampler has explicit control of the agent's state to allow for maximum control of sampling schemes. For comparison, rlpyt mandated specialized alternating agents for alternating sampling.
- Functionality from various locations in the codebase (e.g. collectors, algorithms, models) that carry out simple transformations on the data should be consolidated into the sampler, where it can potentially be parallelized. 

## Design Issues

- The TrajInfo is currently handled by the cage, where agent info (e.g. value estimates, etc.) are not available. Is this an issue?
    - The purpose of TrajInfo is just to be able to collect episodic information, which is not available from the samples buffer (because episodes are interrupted by batch boundaries)
- How can we abstract how agents/models are shared across processes? Add a `models()` method to the agent for the handler to access all the models that need to be shared.
- On reset, the `previous_action` and `previous_reward` seen by the agent (e.g. zeroes) is not the same as what is saved into the samples buffer (e.g. the last action and reward of the previous episode). This is not the case for the observation, where the last observation "after" the episode is simply overwritten by the reset observation of the next episode. For this reason, we need to maintain dedicated "step" variables for action and reward, which are copied into the samples buffer. Can we solve this in a way that is cleaner/nicer?
    In rlpyt, as a result of this, the rlpyt samples buffer is not strictly correct after resets, because the `previous_action` and `previous_reward` are not zeroed out. However, this does not matter because only recurrent agents use these previous values, and these agents do not reset mid-batch.
- Buffer registration and reduction is very similar to `mp.shared_memory.SharedMemory`. If looking up SharedMemory in the global registry is not too slow, could we just rely on this functionality instead? However, the benefit of the buffer registry is that deeply nested namedarraytuples are recovered in a single step.

## TODOs
- Implement (optional) out parameter for agent.step and cage.step methods. This reduces copying but also solves the problem of efficiently handling (parallel) write operations while keeping control in the sampler.
- Rename buffers to Arrays (or something else) (look at tree structures in JAX for inspiration). Nomenclature:
    - Buffer is a (potentially nested) tuple/namedtuple/namedarraytuple of arrays
    - Arrays are either numpy ndarrays, (or a subclass) or torch tensors
- Establish clear interface to Handler/Agent - is the `obs + prev_action` or `obs + prev_action + prev_reward`? This also sort of makes the mini sampler obsolete, unless it hard-codes `(None)` for `prev_action + prev_reward`.
- Clean up cage logic. Use already_done to prevent extra step calls. Reset is available as sync and async calls. Add optional auto-reset mode for performance.
- Samples transformations, e.g. reward normalization, observation normalization, advantage estimation (jitted), creation of `valid` array
- NamedArrayTuple/NamedTuple `__repr__` method should return a dict for easier debug viewing.
- Add simple interface to Stable Baselines in the form of a gym wrapper that looks like the parallel vector wrapper but preallocates memory.

## Ideas
- Sampler types:
    - RecurrentSampler, which samples `prev_action` and `prev_reward` and waits to reset between batches. This sampler supports training recurrent agents with Pytorch.
    - FeedForwardSampler, which does not sample previous values and resets on demand. This only supports feed-forward agents but should be a little faster and use less memory.
    - AlternatingSampler, which alternates stepping half of the envs at a time to provide better performance for slow environments.
    - FullEpisodeSampler, which returns only completed trajectories every iteration. This is essentially a configuration of the RecurrentSampler (cages wait to reset, sampler stops if all envs done, samples buffer allocated with T equal to maximum episode length). Depending on wait-reset semantics, it might not make sense to have a separate class for this.
    - StrictlyCorrectSampler (better name required), which saves the final observation in a trajectory and also contains null values for `prev_action` and `prev_reward` for the first steps in a trajectory, even when not at the beginning of a batch.
        - NOTE: rlpyt sampling does not correctly produce `prev_action` and `prev_reward` if `WaitResetCollector` types are not used. It assumes that these inputs are only used by recurrent agents when using `WaitResetCollectors`, in which case resets only occur at the beginning of a batch.
    - valid array should be created in sampler, not in algo. This will be part of the env samples buffer, preserving a consistent return structure of the samplers
- Array types:
    - AlternatingArray wraps 2 arrays which alternate being written to (by the sampler) and read from (by the algorithm).
- NamedArrayTuples:
    - create metaclass to override behaviour of `type()`, so that getting the type of a Named\[Array\]Tuple returns the corresponding Named\[Array\]TupleClass. This enables code like `type(named_tup)(*iterable)`
        - This keeps NamedArrayTuple picklable, but allows it to behave like a normal type.
    - dictionaries can be written directly into namedarraytuples, e.g.:
        ```
        env_info_buffer[0,5] = env_info_dict
        ```
        - this should eliminate the need for an environment wrapper. There may be some asserts required during buffer creation, to ensure that e.g. the reward is the right data type, etc.
- Handler is responsible for converting buffer structures to pytorch/jax. Instead of converting at each step, the entire samples buffer is converted on init and indexed into at each step.
