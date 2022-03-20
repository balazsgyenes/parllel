from typing import List, Sequence, Tuple

import numpy as np

from parllel.arrays.array import Array
from parllel.buffers import buffer_func, buffer_replace
from parllel.cages import Cage
from parllel.handlers import Handler
from parllel.transforms import Transform
from parllel.types import BatchSpec, TrajInfo
from .collections import Samples

class MiniSampler:
    """Generates a batch of samples, where environments that are done are reset
    immediately. Use this sampler for non-recurrent agents.
    """
    def __init__(self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        agent: Handler,
        batch_buffer: Samples,
        get_bootstrap_value: bool = False,
        batch_transform: Transform = None,
    ) -> None:
        self.batch_spec = batch_spec
        self.get_bootstrap_value = get_bootstrap_value
        
        if batch_transform is None:
            batch_transform = lambda x: x
        self.batch_transform = batch_transform

        self.agent = agent
        self.envs = tuple(envs)
        assert len(self.envs) == self.batch_spec.B
        self.batch_buffer = batch_buffer

        self._init_buffer_transform()

        # bring all environments into a known state
        self.reset_all()

    def reset_all(self) -> None:
        """Reset all environments and save to beginning of observation
        buffer.
        """
        observation = self.batch_buffer.env.observation
        for b, env in enumerate(self.envs):
            # save reset observation to the end of buffer, since it will be 
            # rotated to the beginning
            env.reset_async(out_obs=observation[observation.last + 1, b])

        # reset RNN state of agent, if any
        self.agent.reset()

        # wait for envs to finish reset
        for b, env in enumerate(self.envs):
            env.await_step()

    def _init_buffer_transform(self) -> None:
        # initialize transforms using a dry run
        batch_samples = buffer_func(np.asarray, self.batch_buffer)
        transformed_samples = self.batch_transform.dry_run(batch_samples)

        # the result may include additional buffer elements, so keep returned
        # namedarraytuple, but replace with original Array objects
        self.batch_buffer = buffer_replace(transformed_samples, self.batch_buffer)

        # convert newly-allocated buffer elements to Array objects
        def numpy_to_array(element):
            if isinstance(element, np.ndarray):
                return Array.from_numpy(element)
            return element
        self.batch_buffer = buffer_func(numpy_to_array, self.batch_buffer)

    def get_example_output(self) -> Samples:
        """Get example of a batch of samples."""
        # we can't guarantee that there will be any completed trajectories, so
        # we just ignore them
        example_batch, _ = self.collect_batch(0)

        self.reset_all()

        return example_batch

    def decorrelate_environments(self) -> None:
        """Randomly step environments so they are not all synced up."""
        # TODO: model this off of sampling loop
        pass

    def collect_batch(self, elapsed_steps: int) -> Tuple[Samples, List[TrajInfo]]:
        # get references to buffer elements
        action, agent_info = (
            self.batch_buffer.agent.action,
            self.batch_buffer.agent.agent_info,
        )
        observation, reward, done, env_info = (
            self.batch_buffer.env.observation,
            self.batch_buffer.env.reward,
            self.batch_buffer.env.done,
            self.batch_buffer.env.env_info,
        )

        # rotate last values from previous batch to become previous values
        observation.rotate()

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)
        
        # main sampling loop
        for t in range(self.batch_spec.T):
            # agent observes environment and outputs actions
            # step_action and step_reward are from previous time step (t-1)
            self.agent.step(observation[t], out_action=action[t],
                out_agent_info=agent_info[t])

            for b, env in enumerate(self.envs):
                env.step_async(action[t, b],
                    out_obs=observation[t+1, b], out_reward=reward[t, b],
                    out_done=done[t, b], out_info=env_info[t, b])

            for b, env in enumerate(self.envs):
                env.await_step()

            # if environment is done, reset agent
            # environment has already been reset inside cage
            for b, env in enumerate(self.envs):
                if done[t, b]:
                    self.agent.reset_one(env_index=b)

        if self.get_bootstrap_value:
            # get bootstrap value for last observation in trajectory
            self.batch_buffer.agent.bootstrap_value[:] = self.agent.value(
                observation[self.batch_spec.T])

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj for env in self.envs for traj
            in env.collect_completed_trajs()
            ]

        # convert to underlying numpy array
        batch_samples = buffer_func(np.asarray, self.batch_buffer)

        batch_samples = self.batch_transform(batch_samples)

        return batch_samples, completed_trajectories

    def close(self) -> Tuple[Sequence[Cage], Handler, Samples]:
        return self.envs, self.agent, self.batch_buffer
