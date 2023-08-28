from __future__ import annotations

from typing import Sequence

import numpy as np

from parllel import Array, ArrayDict
from parllel.agents import Agent
from parllel.cages import Cage, TrajInfo
from parllel.transforms import StepTransform, Transform
from parllel.types import BatchSpec

from .sampler import Sampler


class EvalSampler(Sampler):
    def __init__(
        self,
        max_traj_length: int,
        max_trajectories: int,
        envs: Sequence[Cage],
        agent: Agent,
        sample_tree: ArrayDict[Array],
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
            sample_tree=sample_tree,
        )

        self.max_traj_length = max_traj_length
        self.max_trajectories = max_trajectories
        self.obs_transform = obs_transform

    def collect_batch(self, elapsed_steps: int) -> list[TrajInfo]:
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

        # set agent to eval mode, preventing sampler states from being overwritten
        self.agent.eval_mode(elapsed_steps)

        # reset all environments and agent recurrent state
        self.reset()

        # rotate reset observations to be current values
        sample_tree.rotate()

        # TODO: freeze statistics in obs normalization

        n_completed_trajs = 0

        # main sampling loop
        for t in range(self.max_traj_length):
            # apply any transforms to the observation before the agent steps
            if self.obs_transform is not None:
                sample_tree = self.obs_transform(sample_tree, 0)

            # agent observes environment and outputs actions
            action[...], agent_info[...] = self.agent.step(observation[0])

            for b, env in enumerate(self.envs):
                env.step_async(
                    action[0, b],
                    out_next_obs=next_observation[0, b],
                    out_obs=observation[0, b],
                    out_reward=reward[0, b],
                    out_terminated=terminated[0, b],
                    out_truncated=truncated[0, b],
                    out_info=env_info[0, b],
                )

            for b, env in enumerate(self.envs):
                env.await_step()

            done[:] = np.logical_or(terminated, truncated)

            # if environment is done, reset agent
            # environment has already been reset inside cage
            if np.any(done):
                n_completed_trajs += np.sum(done)
                if n_completed_trajs >= self.max_trajectories:
                    break
                self.agent.reset_one(np.asarray(done))

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj for env in self.envs for traj in env.collect_completed_trajs()
        ]

        return completed_trajectories
