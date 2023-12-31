from __future__ import annotations

from typing import Sequence

import numpy as np

from parllel import Array, ArrayDict
from parllel.agents import Agent
from parllel.cages import Cage, TrajInfo
from parllel.transforms import Transform
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
        sample_tree: ArrayDict[Array],
        max_steps_decorrelate: int | None = 0,
        get_bootstrap_value: bool = False,
        step_transforms: Sequence[Transform] | None = None,
        batch_transforms: Sequence[Transform] | None = None,
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
            sample_tree=sample_tree,
            max_steps_decorrelate=max_steps_decorrelate,
        )

        if get_bootstrap_value and "bootstrap_value" not in sample_tree:
            raise ValueError(
                "Expected the sample tree to have a `bootstrap_value` element. Please allocate it."
            )
        self.get_bootstrap_value = get_bootstrap_value

        self.step_transforms = step_transforms if step_transforms is not None else []
        self.batch_transforms = batch_transforms if batch_transforms is not None else []

        # prepare cages and agent for sampling
        self.reset()

    def collect_batch(
        self,
        elapsed_steps: int,
    ) -> tuple[ArrayDict[Array], list[TrajInfo]]:
        # get references to sample tree elements
        action = self.sample_tree["action"]
        agent_info = self.sample_tree["agent_info"]
        next_observation = self.sample_tree["next_observation"]
        observation = self.sample_tree["observation"]
        reward = self.sample_tree["reward"]
        done = self.sample_tree["done"]
        terminated = self.sample_tree["terminated"]
        truncated = self.sample_tree["truncated"]
        env_info = self.sample_tree["env_info"]
        sample_tree = self.sample_tree

        # rotate last values from previous batch to become previous values
        sample_tree.rotate()

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)

        # main sampling loop
        for t in range(self.batch_spec.T):
            # apply any transforms to the observation before the agent steps
            for transform in self.step_transforms:
                # apply in-place to avoid redundant array write operation
                transform(sample_tree[t])

            # agent observes environment and outputs actions
            action[t], agent_info[t] = self.agent.step(observation[t])

            for b, env in enumerate(self.envs):
                env.step_async(
                    action[t, b],
                    out_next_obs=next_observation[t, b],
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
            sample_tree["bootstrap_value"][...] = self.agent.value(
                self.sample_tree["observation"][self.batch_spec.T]
            )

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj for env in self.envs for traj in env.collect_completed_trajs()
        ]

        # apply user-defined transforms
        for transform in self.batch_transforms:
            sample_tree = transform(sample_tree)

        return sample_tree, completed_trajectories
