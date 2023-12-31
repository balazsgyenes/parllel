from __future__ import annotations

from typing import Sequence

import numpy as np

from parllel import Array, ArrayDict
from parllel.agents import Agent
from parllel.cages import Cage, TrajInfo
from parllel.transforms import Transform
from parllel.types import BatchSpec

from .sampler import Sampler


class RecurrentSampler(Sampler):
    def __init__(
        self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        agent: Agent,
        sample_tree: ArrayDict[Array],
        max_steps_decorrelate: int | None = 0,
        get_initial_rnn_state: bool = True,
        get_bootstrap_value: bool = False,
        step_transforms: Sequence[Transform] | None = None,
        batch_transforms: Sequence[Transform] | None = None,
    ) -> None:
        """Generates samples for training recurrent agents."""
        super().__init__(
            batch_spec=batch_spec,
            envs=envs,
            agent=agent,
            sample_tree=sample_tree,
            max_steps_decorrelate=max_steps_decorrelate,
        )

        for cage in envs:
            if cage.reset_automatically:
                raise ValueError(
                    "Expected cages with`reset_automatically=False` for recurrent sampling."
                )

        # verify that valid field exists
        if "valid" not in sample_tree:
            raise ValueError(
                "Expected the sample tree to have a `valid` element. Please allocate it."
            )

        # verify that valid has padding >= 1
        try:
            # try writing beyond the apparent bounds of the action array tree
            sample_tree["valid"][batch_spec.T] = 0
        except IndexError:
            raise ValueError("sample_tree[`valid`] must have padding >= 1")

        # verify that initial_rnn_state field exists
        if get_initial_rnn_state and "initial_rnn_state" not in sample_tree:
            raise ValueError(
                "Expected the sample tree to have a `initial_rnn_state` element. Please allocate it."
            )
        self.get_initial_rnn_state = get_initial_rnn_state

        if get_bootstrap_value and "bootstrap_value" not in self.sample_tree:
            raise ValueError(
                "Expected the sample tree to have a `bootstrap_value` element. Please allocate it."
            )
        self.get_bootstrap_value = get_bootstrap_value

        self.step_transforms = step_transforms if step_transforms is not None else []
        self.batch_transforms = batch_transforms if batch_transforms is not None else []

        # prepare cages for sampling
        self.reset()

    def collect_batch(
        self,
        elapsed_steps: int,
    ) -> tuple[ArrayDict[Array], list[TrajInfo]]:
        # get references to sample tree elements
        action = self.sample_tree["action"]
        agent_info = self.sample_tree["agent_info"]
        observation = self.sample_tree["observation"]
        reward = self.sample_tree["reward"]
        done = self.sample_tree["done"]
        terminated = self.sample_tree["terminated"]
        truncated = self.sample_tree["truncated"]
        env_info = self.sample_tree["env_info"]
        valid = self.sample_tree["valid"]
        sample_tree = self.sample_tree

        # rotate last values from previous batch to become previous values
        sample_tree.rotate()

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)

        if self.get_initial_rnn_state:
            sample_tree["initial_rnn_state"][...] = self.agent.initial_rnn_state()

        # first time step is always valid, rest are invalid by default
        valid[0] = True
        valid[1:] = False
        # other fields do not need to be cleared to 0, because they are either
        # overwritten by fresh data or remain invalid

        # main sampling loop
        envs_to_step = list(enumerate(self.envs))
        for t in range(self.batch_spec.T):
            # get a list of environments that are not done yet
            # we want to avoid stepping these
            envs_to_step = [(b, env) for b, env in envs_to_step if not env.needs_reset]

            if not envs_to_step:
                # all done, we can stop sampling now
                break

            # apply any transforms to the observation before the agent steps
            for transform in self.step_transforms:
                # apply in-place to avoid redundant array write operation
                transform(sample_tree[t])

            # agent observes environment and outputs actions
            action[t], agent_info[t] = self.agent.step(observation[t])

            for b, env in envs_to_step:
                env.step_async(
                    action[t, b],
                    out_obs=observation[t + 1, b],
                    out_reward=reward[t, b],
                    out_terminated=terminated[t, b],
                    out_truncated=truncated[t, b],
                    out_info=env_info[t, b],
                )

            for b, env in envs_to_step:
                env.await_step()

            # calculate validity of samples in next time step
            # this might be required by the obs_transform
            done[t] = np.logical_or(terminated[t], truncated[t])
            valid[t + 1] = np.logical_and(valid[t], np.logical_not(done[t]))

        if self.get_bootstrap_value:
            # get bootstrap value for last observation in trajectory
            # if environment is already done, this value is invalid, but then
            # it will be ignored anyway
            sample_tree["bootstrap_value"][...] = self.agent.value(
                self.sample_tree["observation"][self.batch_spec.T]
            )

        # reset any environments that need reset in parallel
        envs_need_reset = [
            (b, env) for b, env in enumerate(self.envs) if env.needs_reset
        ]

        for b, env in envs_need_reset:
            # overwrite next first observation with reset observation
            env.reset_async(
                out_obs=observation[self.batch_spec.T, b],
            )

        self.agent.reset_one([b for b, env in envs_need_reset])

        for b, env in envs_need_reset:
            env.await_step()

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj for env in self.envs for traj in env.collect_completed_trajs()
        ]

        # apply user-defined transforms
        for transform in self.batch_transforms:
            sample_tree = transform(sample_tree)

        return sample_tree, completed_trajectories
