from typing import List, Optional, Sequence, Tuple

import numpy as np

from parllel.buffers.utils import buffer_map, buffer_rotate
from parllel.cages import Cage
from parllel.handlers import Handler
from parllel.transforms import Transform
from parllel.types import BatchSpec, TrajInfo

from .collections import Samples
from .sampler import Sampler


class BasicSampler(Sampler):
    """Generates a batch of samples, where environments that are done are reset
    immediately. Use this sampler for non-recurrent agents.
    """
    def __init__(self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        agent: Handler,
        batch_buffer: Samples,
        max_steps_decorrelate: Optional[int] = None,
        get_bootstrap_value: bool = False,
        obs_transform: Transform = None,
        batch_transform: Transform = None,
    ) -> None:
        for cage in envs:
            if cage.wait_before_reset:
                raise ValueError("BasicSampler expects cages that reset"
                    " environments immediately. Set wait_before_reset=False")
        
        super().__init__(
            batch_spec = batch_spec,
            envs = envs,
            agent = agent,
            batch_buffer = batch_buffer,
            max_steps_decorrelate = max_steps_decorrelate,
        )

        self.get_bootstrap_value = get_bootstrap_value
        
        if obs_transform is None:
            obs_transform = lambda x, t: x
        self.obs_transform = obs_transform

        if batch_transform is None:
            batch_transform = lambda x: x
        self.batch_transform = batch_transform

        # prepare cages for sampling
        self.initialize()

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
        buffer_rotate(self.batch_buffer)

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)
        
        # main sampling loop
        for t in range(self.batch_spec.T):

            # apply any transforms to the observation before the agent steps
            self.batch_samples = self.obs_transform(self.batch_buffer, t)

            # agent observes environment and outputs actions
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

        batch_samples = self.batch_transform(self.batch_buffer)

        # convert to underlying numpy array
        batch_samples = buffer_map(np.asarray, batch_samples)

        return batch_samples, completed_trajectories
