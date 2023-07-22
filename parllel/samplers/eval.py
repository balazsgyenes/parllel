from __future__ import annotations

from typing import Sequence

import numpy as np

from parllel import Array, ArrayDict
from parllel.cages import Cage, TrajInfo
from parllel.handlers import Agent
from parllel.transforms import StepTransform, Transform
from parllel.types import BatchSpec

from .sampler import Sampler


class EvalSampler(Sampler):
    def __init__(
        self,
        max_traj_length: int,
        min_trajectories: int,
        envs: Sequence[Cage],
        agent: Agent,
        step_buffer: ArrayDict[Array],
        obs_transform: StepTransform | Transform | None = None,
    ) -> None:
        for cage in envs:
            if not cage.reset_automatically:
                raise ValueError(
                    "EvalSampler expects cages that reset environments "
                    "automatically. Set `reset_automatically=True`."
                )

        super().__init__(
            batch_spec=BatchSpec(1, len(envs)),
            envs=envs,
            agent=agent,
            sample_buffer=step_buffer,
        )

        self.max_traj_length = max_traj_length
        self.min_trajectories = min_trajectories
        self.obs_transform = obs_transform

    def collect_batch(self, elapsed_steps: int) -> list[TrajInfo]:
        # get references to buffer elements
        action = self.sample_buffer["action"]
        agent_info = self.sample_buffer["agent_info"]
        observation = self.sample_buffer["observation"]
        reward = self.sample_buffer["reward"]
        done = self.sample_buffer["done"]
        terminated = self.sample_buffer["terminated"]
        truncated = self.sample_buffer["truncated"]
        env_info = self.sample_buffer["env_info"]
        sample_buffer = self.sample_buffer

        # set agent to eval mode, preventing sampler states from being overwritten
        self.agent.eval_mode(elapsed_steps)

        # reset all environments and agent recurrent state
        self.reset()

        # rotate reset observations to be current values
        sample_buffer.rotate()

        # TODO: freeze statistics in obs normalization

        n_completed_trajs = 0

        # main sampling loop
        for t in range(self.max_traj_length):
            # apply any transforms to the observation before the agent steps
            if self.obs_transform is not None:
                sample_buffer = self.obs_transform(sample_buffer, 0)

            # agent observes environment and outputs actions
            action[...], agent_info[...] = self.agent.step(observation[0])

            for b, env in enumerate(self.envs):
                env.step_async(
                    action[0, b],
                    out_obs=observation[0, b],
                    out_reward=reward[0, b],
                    out_done=done[0, b],
                    out_terminated=terminated[0, b],
                    out_truncated=truncated[0, b],
                    out_info=env_info[0, b],
                )

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
            traj for env in self.envs for traj in env.collect_completed_trajs()
        ]

        return completed_trajectories
