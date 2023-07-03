## Design Goals

rlpyt is a great piece of software, but there are several pain points when it comes to working with it.
- Mixing of code for RL and code for efficient utilization of hardware, whereas a typical workflow would involve getting an RL algorithm to function before attempting to accelerate it.
- Certain types have to be used together (e.g. GpuSampler and GpuCollectors)
- Complicated inheritance hierarchies with mixins, especially for agents.
- Duplicated code for cases that share a lot of similarity, especially for collectors, samplers.


## Design Decisions

- Add dependency injection to increase composability of different types. As much as possible, objects should not be responsible for initializing their dependent objects (e.g. sampler should not have to initialize collectors, agent, etc.). The initialization process is handled in the top-level script, assisted by patterns, functions that act as shortcuts for common use cases.
- The user should be able to easily control the contents of the sampler buffer. For this reason, the sampler should be kept as simple as possible to facilitate the writing of new sampler classes. Out arguments (e.g. `out_observation`) make reading and writing data transparent.
- Initialization takes as input an example of the input the object will see during the training loop, and return an example of its output. This allows all objects to pre-allocate memory and data types for efficiency, without hard-coding dependencies.
- Code for RL and code for optimization should remain separated. Instead of combining these objects through inheritance, one should wrap the other (e.g. Handler wraps Agent instead of handler code being in base class of Agent).
- Sampler has explicit control of the agent's state to allow for maximum control of sampling schemes. For comparison, rlpyt mandated specialized alternating agents for alternating sampling.
- Functionality from various locations in the codebase (e.g. collectors, algorithms, models) that carry out simple transformations on the data should be consolidated into the sampler, where it can potentially be parallelized.
- Buffer registration allows deeply-nested buffer structures to be sent quickly over pipes, since the top-level buffer is also in the registry. This is in contrast with `mp.shared_memory.SharedMemory`, where only the arrays at the leaves of the buffer are pickled by name.

## Sharp Edges
- Writing `named_array_tuple[:] = None` silently does nothing. This can occur e.g. if expecting Cage to return a value but `out` parameters were passed to the function.
- In many cases, an array (or buffer) can be a `RotatingArray` or non-RotatingArray, for which indexing at -1 silently results in different behaviour. Indexing at -1 is therefore discouraged; instead index at `array.last`, produces the expected result in all cases.

## Long-Term Design Issues

- The TrajInfo is currently handled by the cage, where agent info (e.g. value estimates, etc.) are not available. Is this an issue?
    - The purpose of TrajInfo is just to be able to collect episodic information, which is not available from the samples buffer (because episodes are interrupted by batch boundaries)
- How can we abstract how agents/models are shared across processes? Add a `models()` method to the agent for the handler to access all the models that need to be shared.
- Is there any way to recover the final observation from a trajectory, after the env is done but before reset? Is there any use case for this?
- Handler's main job is to handle out_args, but it's not clear if this is needed. If this is removed, then torchifying buffers can become the responsibility of the agent. In this case, handler would be optional and base handler would be deleted (torch handler only, if at all). Also, handler wouldn’t appear in most type hints (e.g. in the sampler).
- Can we avoid defining a rigid interface to the Handler/Agent/Model? How can a user add another argument to the agent/model, and what is the use case for this? (e.g. agent_ids for multi-agent case)
- What object(s) are responsible for array allocation for the batch buffer. This batch buffer is basically global state, so it falls under the responsibility of the build function, but Transform types currently allocate their own additional Arrays.
- With complete control of the pipe and pickler, we should be able to make buffer registration a lot more seamless. Can we implement pytorch's approach, where arrays are automatically copied into shared memory (and registered) the first time they are moved between processes?
- Should objects strictly only receive constructor arguments that they need during execution? This sounds like good abstraction, but in practice it means we get a pattern function for every type to handle the "front-end" value processing. e.g. SAC receives replay ratio, sampler batch spec and replay ratio batch size and calculates its own updates_per_optimize. In theory, SAC should just receive updates_per_optimize as an argument, but then this value would have to be calculated by a pattern function.

## TODOs and Ideas

- Top-level functionality:
    - Seeding
        - Add seeding module that maintains a SeedSequence and spawns a new seed sequence each time an entity requests a new seed. Seeds are saved by name so they can be reloaded. Seeds are logged to a json file.
        - If a seed is requested twice for the same name, an error is thrown
    - Logging
        - Log random seeds to enable repeatable runs
        - Enable loading random seeds from log to repeat a previous run
        - Log mean and std_dev of obs normalization, std_dev of reward normalization
        - Add rest of program state (e.g. optimizer state, transform states, etc.) to checkpoints
        - Enable resuming run from checkpoint
        - Add independent WandB log writer so that tensorboard becomes optional
        - Add additional serializers for config files
            - Maybe use pickle to serialize types in config (but needs to be detectable on read)
            - Maybe use cloudpickle like SB3 does
            - If serializing to yaml, ensure no collisions with wandb's config.yaml file
        - Add vectorized rendering of environments without pixel observations, using a rendering schedule. Ensure cage correctly calls render on gymnasium environment.
    - **!!** Allocators
        - Add allocators module with user-configurable and default logic for what Array type should be used for what buffer element
        - Set parallel attribute in allocator module and then get default CageCls everywhere else
    - Callbacks?
- Algos
    - Handle termination and truncation differently, e.g. for advantage computation and in SAC
    - DDPG
    - Jax PPO :)
- Arrays
    - Add `begin` and `end` attributes, where `end` is intended to be used in a slice
    - Rename `rotate` to something like `next_iteration`
    - Enforce that `-1` is never used to index the last element, i.e. `-1` is never passed to the underlying `ndarray`.
    - SwitchingArray wraps two arrays and switches between them on `rotate`. This is useful for asynchronous sampling, where different parts of the array are simultaneously written to by the sampler and read from by the algorithm.
        - Overloading `rotate` is useful because the batch buffer is already rotated before each batch.
        - SwitchingRotatingArray needs to add 4x padding or maybe we should just allocate 2 arrays.
    - implement `previous` and `next` for `Array` objects with indexing history, returning an array of the same shape but with the time index shifting backward or forward by one, respectively
    - `LazyFramesArray`, which only saves the most recent frame in a LazyFrames object, and recreates the frame stack in its `__array__` method
    - Array equality check verifies that buffer ids and (internal) current_indices are the same (because indices in standard form should be equivalent). This allows the `SynchronizedProcessCage` to check whether the expected array slice was passed.
- Buffers
    - `buffer_get_attr` and `buffer_set_attr`
    - NamedArrayTuple/NamedTuple `__repr__` method should return a dict for easier debug viewing.
    - **!!** Replace NamedTuple and NamedArrayTuple with ArrayDict based on TensorDict.
- Cages:
    - Add `__getattr__`, `__setattr__`, and `env_method` methods to Cage, allowing direct access to env.
    - If `set_samples_buffer` is called on a SharedMemoryArray, it verifies that the buffers are registered before sending the reduced buffer across the pipe. This allows for consistent use in all cases, and supports configurations like a replay buffer in shared memory with ProcessCage.
    - `SynchronizedProcessCage`, where a single Event object is shared among multiple Cages, such that all begin stepping as soon as one of them is called to step. Based on how Events are shared, this supports alternating sampling too.
        - The array slices passed to `async_step` are not sent to the child process, but the parent process verifies that they are the expected ones.
        - When `set_samples_buffer` is called on the `SynchronizedProcessCage`, it saves the array slice that it should iterate through during sampling. Since the array slice is also sent on each time step, other Cage types can ignore it.
    - VectorizedCage, similar to VecEnc in StableBaselines3, which allows for multiple environments in a single process.
    - Add argument to `ProcessCage` to choose between process creation methods
- Handler:
    - Make Handler a subclass of Agent to prevent silly linter problems and redundant wrapper functions.
    - Implement CPU sampling by duplicating the agent with a model parameters on the CPU. Handler overrides `sampling_mode` and `training_mode`, etc. to know when to sync model parameters.
- Runners
    - Add `ChainRunner`, which chains multiple Runners to execute in sequence. `ChainRunner` must implement some resource management, such that resources for a runner are not created until needed, and destroyed after they are no longer needed. This is difficult because it's likely that some resources are shared between runners.
- Samplers
    - All samplers call `set_samples_buffer` on the cages at least once (AsynchronousSampler calls before each batch), where each cage is passed the Array slice where it should write. 
    - AlternatingSampler, which alternates stepping half of the envs at a time to provide better performance for slow environments.
    - AsynchronousSampler, which samples in a child process while the algorithm is optimizing.
        - This class should inherit from the standard sampler types much like how ProcessCage inherits from Cage.
        - Calls `set_samples_buffer` on cages before each batch, so that they write to the correct buffer
- Transforms
    - Add `freeze` method to stop statistics from being updated when evaluating agent.
    - Add `stats` attribute to normalizing transforms (and anything else in the transform that is stateful and could be logged)
    - Add test for `NormalizeObservation` transformation, which verifies that environments that are already done are not factored into running statistics.
- Misc
    - Add simple interface to Stable Baselines in the form of a gym wrapper that looks like the parallel vector wrapper but preallocates memory.


## Bugs

- BUG: CartPole with pixel observations hangs when using headless rendering in parallel environments without the SubprocessWrapper.
- BUG: fix memory leak when using `fork` start method and ManagedMemoryArrays ( https://bugs.python.org/issue38119 )

## To benchmark
- Reading/writing to arrays in shared/managed memory vs. local memory
- Filtering observations before calling `agent.step()` to remove those from envs that are already done
- Lazy indexing of numpy arrays within `Array`. This operation is only on the critical path when there are many envs and the main process must send each of them a slice. For the `SynchronizedProcessCage` this is no longer a problem, because all environments begin stepping as soon as `step_async` is called on the first env, while the parent process continues slicing and verifying that the slices are as expected.
- Performance loss of maintaining separate array for `previous_action` under `agent_info`.
- Performance gain of only saving `initial_rnn_state` to batch buffer
- Performance of indexing cuda tensor with numpy array vs. cpu tensor vs. cuda tensor
