from typing import List, Optional, Sequence, Tuple

import numpy as np

from parllel.buffers import Samples, buffer_asarray
from parllel.buffers.utils import buffer_rotate
from parllel.cages import Cage, TrajInfo
from parllel.handlers import Handler
from parllel.transforms import Transform
from parllel.types import BatchSpec

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

        if get_bootstrap_value and not hasattr(batch_buffer.agent,
                "bootstrap_value"):
            raise ValueError("Bootstrap value is written to batch_buffer.agent"
                ".bootstrap_value, but this field does not exist. Please "
                "allocate it.")
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

        last_T = self.batch_spec.T - 1

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
            if np.any(dones := done[t]):
                self.agent.reset_one(np.asarray(dones))
        
        if self.get_bootstrap_value:
            # get bootstrap value for last observation in trajectory
            self.agent.value(
                observation[last_T + 1],
                out_value=self.batch_buffer.agent.bootstrap_value,
            )

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj
            for env in self.envs
            for traj in env.collect_completed_trajs()
        ]

        batch_samples = self.batch_transform(self.batch_buffer)

        # convert to underlying numpy array
        # TODO: remove this, sampler should return Array objects for flexibility
        batch_samples = buffer_asarray(batch_samples)

        return batch_samples, completed_trajectories
