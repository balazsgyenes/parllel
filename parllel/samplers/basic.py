from __future__ import annotations

from typing import Sequence

import numpy as np

from parllel import Array, ArrayDict
from parllel.agents import Agent
from parllel.cages import Cage, TrajInfo
from parllel.transforms import BatchTransform, StepTransform
from parllel.types import BatchSpec

from .sampler import Sampler


class BasicSampler(Sampler):
    """Generates a batch of samples, where environments that are done are reset
    immediately. Use this sampler for non-recurrent agents.
    """

    def __init__(
        self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        agent: Agent,
        sample_buffer: ArrayDict[Array],
        max_steps_decorrelate: int | None = None,
        get_bootstrap_value: bool = False,
        obs_transform: StepTransform | None = None,
        batch_transform: BatchTransform | None = None,
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

        if get_bootstrap_value and "bootstrap_value" not in sample_buffer:
            raise ValueError(
                "Expected the sample buffer to have a `bootstrap_value` element. Please allocate it."
            )
        self.get_bootstrap_value = get_bootstrap_value

        self.obs_transform = obs_transform
        self.batch_transform = batch_transform

        # prepare cages and agent for sampling
        self.reset()

    def collect_batch(
        self,
        elapsed_steps: int,
    ) -> tuple[ArrayDict[Array], list[TrajInfo]]:
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

        # rotate last values from previous batch to become previous values
        sample_buffer.rotate()

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)

        # main sampling loop
        for t in range(self.batch_spec.T):
            # apply any transforms to the observation before the agent steps
            if self.obs_transform is not None:
                sample_buffer = self.obs_transform(sample_buffer, t)

            # agent observes environment and outputs actions
            action[t], agent_info[t] = self.agent.step(observation[t])

            for b, env in enumerate(self.envs):
                env.step_async(
                    action[t, b],
                    out_obs=observation[t + 1, b],
                    out_reward=reward[t, b],
                    out_terminated=terminated[t, b],
                    out_truncated=truncated[t, b],
                    out_info=env_info[t, b],
                )

            for b, env in enumerate(self.envs):
                env.await_step()

            done[t] = np.logical_or(terminated[t], truncated[t])

            # if environment is done, reset agent
            # environment has already been reset inside cage
            if np.any(dones := done[t]):
                self.agent.reset_one(np.asarray(dones))

        if self.get_bootstrap_value:
            # get bootstrap value for last observation in trajectory
            sample_buffer["bootstrap_value"][...] = self.agent.value(
                self.sample_buffer["observation"][self.batch_spec.T]
            )

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj for env in self.envs for traj in env.collect_completed_trajs()
        ]

        # apply user-defined transforms
        if self.batch_transform is not None:
            sample_buffer = self.batch_transform(sample_buffer)

        return sample_buffer, completed_trajectories
