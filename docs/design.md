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
