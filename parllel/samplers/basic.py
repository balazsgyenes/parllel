from typing import List, Optional, Sequence, Tuple

import numpy as np

from parllel.buffers import Samples
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
        sample_buffer: Samples,
        max_steps_decorrelate: Optional[int] = None,
        get_bootstrap_value: bool = False,
        obs_transform: Optional[Transform] = None,
        batch_transform: Optional[Transform] = None,
    ) -> None:
        for cage in envs:
            if not cage.reset_automatically:
                raise ValueError(
                    "BasicSampler expects cages that reset environments "
                    "automatically. Set `reset_automatically=True`."
                )
        
        super().__init__(
            batch_spec=batch_spec,
            envs=envs,
            agent=agent,
            sample_buffer=sample_buffer,
            max_steps_decorrelate=max_steps_decorrelate,
        )

        if get_bootstrap_value and not hasattr(self.sample_buffer.agent,
                "bootstrap_value"):
            raise ValueError("Bootstrap value is written to sample_buffer.agent"
                ".bootstrap_value, but this field does not exist. Please "
                "allocate it.")
        self.get_bootstrap_value = get_bootstrap_value
        
        self.obs_transform = obs_transform
        self.batch_transform = batch_transform

        # prepare cages and agent for sampling
        self.reset()

    def collect_batch(self, elapsed_steps: int) -> Tuple[Samples, List[TrajInfo]]:
        # get references to buffer elements
        action, agent_info = (
            self.sample_buffer.agent.action,
            self.sample_buffer.agent.agent_info,
        )
        observation, reward, done, env_info = (
            self.sample_buffer.env.observation,
            self.sample_buffer.env.reward,
            self.sample_buffer.env.done,
            self.sample_buffer.env.env_info,
        )
        sample_buffer = self.sample_buffer

        # rotate last values from previous batch to become previous values
        buffer_rotate(sample_buffer)

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)
        
        # main sampling loop
        for t in range(self.batch_spec.T):

            # apply any transforms to the observation before the agent steps
            if self.obs_transform is not None:
                sample_buffer = self.obs_transform(sample_buffer, t)

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
                observation[self.batch_spec.T],
                out_value=sample_buffer.agent.bootstrap_value,
            )

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj
            for env in self.envs
            for traj in env.collect_completed_trajs()
        ]

        # apply user-defined transforms
        if self.batch_transform is not None:
            sample_buffer = self.batch_transform(sample_buffer)

        return sample_buffer, completed_trajectories
