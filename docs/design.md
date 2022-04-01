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
- Buffer registration allows deeply-nested buffer structures to be sent quickly over pipes, since the top-level buffer is also in the registry. This is in contrast with `mp.shared_memory.SharedMemory`, where only the arrays at the leaves of the buffer are pickled by name.

## Sharp Edges
- Writing `named_array_tuple[:] = None` silently does nothing. This can occur e.g. if expecting Cage to return a value but `out` parameters were passed to the function.
- In many cases, an array (or buffer) can be a `RotatingArray` or non-RotatingArray, for which indexing at -1 silently results in different behaviour. Indexing at -1 is therefore discouraged; instead index at `array.last`, produces the expected result in all cases.

## Long-Term Design Issues

- In recurrent problems, all agent state should be handled in the same way. rnn_state is outputted by the agent and saved to agent_info. However, previous_action is maintained by the batch buffer as if it were the same as the action. However, previous action differs from action at the beginning of each trajectory, so they are not the same. This is only remedied because only the first time step can be the beginning of a trajectory (in recurrent problems), so this is explicitly handled by the sampler. But strictly speaking, prev_action should be a separate buffer field under agent_info.
    - If we maintained the current solution, sequence replay buffers would need to detect resets. Actions would need to be copied into RotatingArray, and previous_action immediately before a reset would need to be zeroed out.
    - Are rnn_states after the first time step in a batch even needed? They probably take up a fair amount of space. Another solution would be to create a separate batch buffer field for the initial values of agent state (e.g. rnn_state, prev_action), and `agent_info.rnn_state` is otherwise set to None (which disables writes). The remaining state values can then be "reconstructed" in the algorithm.
- The TrajInfo is currently handled by the cage, where agent info (e.g. value estimates, etc.) are not available. Is this an issue?
    - The purpose of TrajInfo is just to be able to collect episodic information, which is not available from the samples buffer (because episodes are interrupted by batch boundaries)
- How can we abstract how agents/models are shared across processes? Add a `models()` method to the agent for the handler to access all the models that need to be shared.
- Is there any way to recover the final observation from a trajectory, after the env is done but before reset? Is there any use case for this?
- Can we avoid defining a rigid interface to the Handler/Agent/Model? How can a user add another argument to the agent/model, and what is the use case for this? (e.g. agent_ids for multi-agent case)
- What object(s) are responsible for array allocation for the batch buffer. This batch buffer is basically global state, so it falls under the responsibility of the build function, but Transform types currently allocate their own additional Arrays.
- With complete control of the pipe and pickler, we should be able to make buffer registration a lot more seamless. Can we implement pytorch's approach, where arrays are automatically copied into shared memory (and registered) the first time they are moved between processes?

## TODOs and Ideas

- Top-level functionality:
    - Seeding
        - Add seeding module that maintains a SeedSequence and spawns a new seed sequence each time an entity requests a new seed. Seeds are saved by name so they can be reloaded. Seeds are logged to a json file.
        - If a seed is requested twice for the same name, an error is thrown
    - Config handling. User passes a giant dictionary of config parameters to a build method
    - Logging of start state (e.g. config, seeds, allocators, etc.) to compare runs after the fact and allow for repeating runs
    - Logging of diagnostic data (e.g. rewards, traj length, etc.) during training to analyze results
        - Data is saved to tensorboard, csv file, and log file
        - `log_array` logs mean, max, min, std_dev, and median of values in the array
        - `log_value` logs a single value as given (e.g. observation mean and std_dev)
    - Logging of notifications and warnings
        - Data is saved to log file and standard output
    - Saving of checkpoints (e.g. model parameters, optimizer state, transform states, etc.) to allow for resuming training
    - Loading of previous runs for e.g. running a policy, repeating a run, resuming a run
    - Allocators
        - Add allocators module with user-configurable and default logic for what Array type should be used for what buffer element
    - Callbacks?
- Algos
    - Any many more methods that subclasses could overwrite (e.g. `construct_agent_inputs` in PPO)
    - Jax PPO :)
- Agents/Distributions
    - Ensemble agent
    - Ensemble distribution
- Arrays
    - **!!** Add `first` and `last` attributes to base `Array` class.
    - SwitchingArray wraps two arrays and switches between them on `rotate`. This is useful for asynchronous sampling, where different parts of the array are simultaneously written to by the sampler and read from by the algorithm.
        - Overloading `rotate` is useful because the batch buffer is already rotated before each batch.
        - SwitchingRotatingArray needs to add 4x padding or maybe we should just allocate 2 arrays.
    - Write algorithm enabling Array types to keep track of their current indices. This value will be internal only, since it will not match the index_history for RotatingArray, but will enable many features:
        - implement `previous` and `next` for `Array` objects with indexing history, returning an array of the same shape but with the time index shifting backward or forward by one, respectively
        - `LazyFramesArray`, which only saves the most recent frame in a LazyFrames object, and recreates the frame stack in its `__array__` method
        - `Array` already abstracts indexing items of arrays, where for numpy arrays this results in a copy. Add support for indexing Array with an array or list of integers without copying (this also results in a copy when used on numpy arrays)
        - Array equality check verifies that buffer ids and (internal) current_indices are the same (because indices in standard form should be equivalent). This allows the `SynchronizedProcessCage` to check whether the expected array slice was passed.
- Buffers
    - `buffer_get_attr` and `buffer_set_attr`
    - NamedArrayTuple/NamedTuple `__repr__` method should return a dict for easier debug viewing.
- Cages:
    - **!!** Actually implement `already_done` in `ProcessCage`.
    - Add `__getattr__`, `__setattr__`, and `env_method` methods to Cage, allowing direct access to env.
    - If `set_samples_buffer` is called on a SharedMemoryArray, it verifies that the buffers are registered before sending the reduced buffer across the pipe. This allows for consistent use in all cases, and supports configurations like a replay buffer in shared memory with ProcessCage.
    - `SynchronizedProcessCage`, where a single Event object is shared among multiple Cages, such that all begin stepping as soon as one of them is called to step. Based on how Events are shared, this supports alternating sampling too.
        - The array slices passed to `async_step` are not sent to the child process, but the parent process verifies that they are the expected ones.
        - When `set_samples_buffer` is called on the `SynchronizedProcessCage`, it saves the array slice that it should iterate through during sampling. Since the array slice is also sent on each time step, other Cage types can ignore it.
    - VectorizedCage, similar to VecEnc in StableBaselines3, which allows for multiple environments in a single process.
    - **??** Add argument to `ProcessCage` to choose between process creation methods
- Handler:
    - Preallocate torch tensor version of batch buffer so that it does not have to be converted at each step (benchmark first to see if this is significant)
    - Implement CPU sampling by spawning an agent with a model parameters on the CPU
- Patterns
    - Include default parameter sets for things like algorithms. Remove all default values in algorithm `__init__` methods.
- Replay buffers
    - Add replay buffers that wrap a samples buffer and expose the ability to sample from it. All required values (e.g. discounted returns) are precomputed by Transforms.
        - Is it possible for the sampler to write directly into the replay buffer, without copying from a smaller batch buffer? This would require calling `set_samples_buffer` before every batch in all cases, since a replay buffer could even be used with BasicSampler.
- Runners
    - Add logging and checkpointing
    - Add `OffPolicyRunner`
    - Add runner that just runs trained policy and renders it
- Samplers
    - **!!** RecurrentSampler, which feeds the agent its `prev_action` and resets environments only between batches.
        - Move `previous_action` into the agent state. Reset it in the agent when `reset_one` is called.
        - Add `previous_action` to the batch buffer and ensure that it is properly zeroed for environments that are reset, e.g. by writing 0 to the last + 1 position at the end of the batch
        - Add mechanism for including `previous_action` in the samples buffer if the agent/algo requires it. Right now it's entirely up to the Sampler what gets passed to the agent, but the algo needs to know this too.
    - Add sampler tests.
        - Also test transform handling in samplers using dummy transforms, and verify that they receive the proper samples in the proper order (e.g. no observation normalization for environments in RecurrentSampler that are already done)
    - All samplers call `set_samples_buffer` on the cages at least once (AsynchronousSampler calls before each batch), where each cage is passed the Array slice where it should write. 
    - AlternatingSampler, which alternates stepping half of the envs at a time to provide better performance for slow environments.
    - AsynchronousSampler, which samples in a child process while the algorithm is optimizing.
        - This class should inherit from the standard sampler types much like how ProcessCage inherits from Cage.
        - Calls `set_samples_buffer` on cages before each batch, so that they write to the correct buffer
    - FullEpisodeSampler, which returns only completed trajectories every iteration. This is essentially a configuration of the RecurrentSampler (cages wait to reset, sampler stops if all envs done, samples buffer allocated with T equal to maximum episode length). Depending on wait-reset semantics, it might not make sense to have a separate class for this.
- Transforms
    - Add `stats` attribute to normalizing transforms (and anything else in the transform that is stateful and could be logged)
    - Multiagent versions of transforms (e.g. advantage estimation)
- Misc
    - Add simple interface to Stable Baselines in the form of a gym wrapper that looks like the parallel vector wrapper but preallocates memory.


## Bugs

- BUG: fix memory leak when using `fork` start method

## To benchmark
- Reading/writing to arrays in shared/managed memory vs. local memory
- Filtering observations before calling `agent.step()` to remove those from envs that are already done
- Lazy indexing of numpy arrays within `Array`. This operation is only on the critical path when there are many envs and the main process must send each of them a slice. For the `SynchronizedProcessCage` this is no longer a problem, because all environments begin stepping as soon as `step_async` is called on the first env, while the parent process continues slicing and verifying that the slices are as expected.
- Performance loss of maintaining separate array for `previous_action` under `agent_info`.
- Performance gain of only saving `initial_rnn_state` to batch buffer
