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


## Long-Term Design Issues

- The TrajInfo is currently handled by the cage, where agent info (e.g. value estimates, etc.) are not available. Is this an issue?
    - The purpose of TrajInfo is just to be able to collect episodic information, which is not available from the samples buffer (because episodes are interrupted by batch boundaries)
- How can we abstract how agents/models are shared across processes? Add a `models()` method to the agent for the handler to access all the models that need to be shared.
- On reset, the `previous_action` and `previous_reward` seen by the agent (e.g. zeroes) is not the same as what is saved into the samples buffer (e.g. the last action and reward of the previous episode). This is not the case for the observation, where the last observation "after" the episode is simply overwritten by the reset observation of the next episode. For this reason, we need to maintain dedicated "step" variables for action and reward, which are copied into the samples buffer. Can we solve this in a way that is cleaner/nicer?
    In rlpyt, as a result of this, the rlpyt samples buffer is not strictly correct after resets, because the `previous_action` and `previous_reward` are not zeroed out. However, this does not matter because only recurrent agents use these previous values, and these agents do not reset mid-batch.
- Buffer registration and reduction is very similar to `mp.shared_memory.SharedMemory`. If looking up SharedMemory in the global registry is not too slow, could we just rely on this functionality instead?
    - However, the benefit of the buffer registry is that deeply nested namedarraytuples are recovered in a single step.
- Can we avoid defining a rigid interface to the Handler/Agent/Model? How can a user add another argument to the agent/model, and what is the use case for this? (e.g. agent_ids for multi-agent case)
- What object(s) are responsible for array allocation for the batch buffer. This batch buffer is basically global state, so it falls under the responsibility of the build function, but Transform types currently allocate their own additional Arrays.


## TODOs

- Seeding!
- Checkpointing and loading checkpoints
- Logging
- Callbacks
- For improving buffer handling, look at JAX tree structures for inspiration.
    - Add Generic type hinting for Buffer, e.g. `Buffer[NDArray]`
- Cages:
    - Add `__getattr__`, `__setattr__`, and `env_method` methods to Cage, allowing direct access to env.
    - Actually implement `already_done` in `ParallelProcessCage`.
    - From Paul: Add calling `set_samples_buffer` on cages to sampler `__init__` method. In parallel sampler, the samples buffer needs to be alternated every batch, so this can be set at each batch.
        - This isn't necessarily what we want, since it makes the sampler responsible for the operation of the cages.
- Step transformations, e.g. observation normalization, image translation
- NamedArrayTuple/NamedTuple `__repr__` method should return a dict for easier debug viewing.
- In Handler, preallocate torch tensor version of batch buffer so that it does not have to be converted at each step.
- Add mechanism for including `previous_action` in the samples buffer if the agent/algo requires it. Right now it's entirely up to the Sampler what gets passed to the agent, but the algo needs to know this too.


## Bugs

- BUG: fix array indexing logic. Is the wrapped array indexed when the Array object is indexed? How can this indexed array state be reconstructed when unpickling? Ensure this is correct in all cases, including rotating arrays.
    - add second index history to rotating array. The public interface returns the unshifted indicies for use by the unpickler in reconstructing the array. The private index history tracks the shifted index history for use by `__setstate__` in reconstructing the numpy array
    - removing `__getstate__` and `__setstate__` might solve this problem as well as result in a performance boost, since they are called by `copy.copy` each time the array is indexed. Can the numpy arrays simply be sent through the pipe?
- BUG: fix memory leak when using `fork` start method


## Ideas

- Sampler types:
    - RecurrentSampler, which samples `prev_action` and `prev_reward` and waits to reset between batches. This sampler supports training recurrent agents with Pytorch.
    - FeedForwardSampler, which does not sample previous values and resets on demand. This only supports feed-forward agents but should be a little faster and use less memory.
    - AlternatingSampler, which alternates stepping half of the envs at a time to provide better performance for slow environments.
    - FullEpisodeSampler, which returns only completed trajectories every iteration. This is essentially a configuration of the RecurrentSampler (cages wait to reset, sampler stops if all envs done, samples buffer allocated with T equal to maximum episode length). Depending on wait-reset semantics, it might not make sense to have a separate class for this.
    - StrictlyCorrectSampler (better name required), which saves the final observation in a trajectory and also contains null values for `prev_action` and `prev_reward` for the first steps in a trajectory, even when not at the beginning of a batch.
        - NOTE: rlpyt sampling does not correctly produce `prev_action` and `prev_reward` if `WaitResetCollector` types are not used. It assumes that these inputs are only used by recurrent agents when using `WaitResetCollectors`, in which case resets only occur at the beginning of a batch.
- Array types:
    - AlternatingArray wraps 2 arrays which alternate being written to (by the sampler) and read from (by the algorithm).
- Prevent calls to `agent.step()` for environments that are done and waiting for be reset. The speedup from this might not be significant.
- Add simple interface to Stable Baselines in the form of a gym wrapper that looks like the parallel vector wrapper but preallocates memory.
- With complete control of the pipe and pickler, we should be able to make buffer registration a lot more seamless. Can we implement pytorch's approach, where arrays are automatically copied into shared memory (and registered) the first time they are moved between processes?
- Add argument to `ParallelProcessCage` to choose between process creation methods
