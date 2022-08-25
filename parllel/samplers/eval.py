from typing import List, Optional, Sequence, Tuple

import numpy as np

from parllel.buffers import Samples
from parllel.buffers.utils import buffer_rotate
from parllel.cages import Cage, TrajInfo
from parllel.handlers import Handler
from parllel.transforms import Transform
from parllel.types import BatchSpec

from .sampler import Sampler


class EvalSampler(Sampler):
    def __init__(self,
        max_traj_length: int,
        min_trajectories: int,
        envs: Sequence[Cage],
        agent: Handler,
        step_buffer: Samples,
        obs_transform: Optional[Transform] = None,
    ) -> None:
        for cage in envs:
            if cage.wait_before_reset:
                raise ValueError("EvaluationSampler expects cages that reset"
                    " environments immediately. Set wait_before_reset=False")
        
        super().__init__(
            batch_spec=BatchSpec(max_traj_length, len(envs)),
            envs=envs,
            agent=agent,
            sample_buffer=step_buffer,
        )

        self.min_trajectories = min_trajectories
        self.obs_transform = obs_transform

        # prepare cages for sampling
        self.reset_envs()

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

        # reset all environments
        self.reset_envs()

        # rotate last values from previous batch to become previous values
        buffer_rotate(sample_buffer)

        # prepare agent for sampling
        self.agent.eval_mode(elapsed_steps)
        self.agent.reset()
        
        # TODO: freeze statistics in obs normalization

        n_completed_trajs = 0

        # main sampling loop
        for _ in range(self.batch_spec.T):

            # apply any transforms to the observation before the agent steps
            if self.obs_transform is not None:
                sample_buffer = self.obs_transform(sample_buffer, 0)

            # agent observes environment and outputs actions
            self.agent.step(observation[0], out_action=action[0],
                out_agent_info=agent_info[0])

            for b, env in enumerate(self.envs):
                env.step_async(action[0, b],
                    out_obs=observation[0, b], out_reward=reward[0, b],
                    out_done=done[0, b], out_info=env_info[0, b])

            for b, env in enumerate(self.envs):
                env.await_step()

            # if environment is done, reset agent
            # environment has already been reset inside cage
            if np.any(dones := done[0]):
                n_completed_trajs += np.sum(dones)
                if n_completed_trajs >= self.min_trajectories:
                    break
                self.agent.reset_one(np.asarray(dones))
        
        # collect all completed trajectories from envs
        completed_trajectories = [
            traj
            for env in self.envs
            for traj in env.collect_completed_trajs()
        ]

        return completed_trajectories
