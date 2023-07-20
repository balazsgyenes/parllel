from __future__ import annotations

from typing import Sequence

import numpy as np

from parllel import Array, ArrayDict
from parllel.cages import Cage, TrajInfo
from parllel.handlers import Agent
from parllel.transforms import BatchTransform, StepTransform
from parllel.types import BatchSpec

from .sampler import Sampler


class RecurrentSampler(Sampler):
    def __init__(
        self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        agent: Agent,
        sample_buffer: ArrayDict[Array],
        max_steps_decorrelate: int | None = None,
        get_initial_rnn_state: bool = True,
        get_bootstrap_value: bool = False,
        obs_transform: StepTransform | None = None,
        batch_transform: BatchTransform | None = None,
    ) -> None:
        """Generates samples for training recurrent agents."""
        super().__init__(
            batch_spec=batch_spec,
            envs=envs,
            agent=agent,
            sample_buffer=sample_buffer,
            max_steps_decorrelate=max_steps_decorrelate,
        )

        for cage in envs:
            if cage.reset_automatically:
                raise ValueError(
                    "Expected cages with`reset_automatically=False` for recurrent sampling."
                )

        # verify that valid field exists
        if "valid" not in sample_buffer:
            raise ValueError(
                "Expected the sample buffer to have a `valid` element. Please allocate it."
            )

        # verify that valid has padding >= 1
        try:
            # try writing beyond the apparent bounds of the action buffer
            sample_buffer["valid"][batch_spec.T] = 0
        except IndexError:
            raise ValueError("sample_buffer[`valid`] must have padding >= 1")

        # verify that initial_rnn_state field exists
        if get_initial_rnn_state and "initial_rnn_state" not in sample_buffer:
            raise ValueError(
                "Expected the sample buffer to have a `initial_rnn_state` element. Please allocate it."
            )
        self.get_initial_rnn_state = get_initial_rnn_state

        if get_bootstrap_value and "bootstrap_value" not in self.sample_buffer:
            raise ValueError(
                "Expected the sample buffer to have a `bootstrap_value` element. Please allocate it."
            )
        self.get_bootstrap_value = get_bootstrap_value

        self.obs_transform = obs_transform
        self.batch_transform = batch_transform

        # prepare cages for sampling
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
        valid = self.sample_buffer["valid"]
        sample_buffer = self.sample_buffer

        # rotate last values from previous batch to become previous values
        sample_buffer.rotate()

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)

        if self.get_initial_rnn_state:
            sample_buffer["initial_rnn_state"][...] = self.agent.initial_rnn_state()

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
            if self.obs_transform is not None:
                self.batch_samples = self.obs_transform(sample_buffer, t)

            # agent observes environment and outputs actions
            self.agent.step(
                observation[t], out_action=action[t], out_agent_info=agent_info[t]
            )

            for b, env in envs_to_step:
                env.step_async(
                    action[t, b],
                    out_obs=observation[t + 1, b],
                    out_reward=reward[t, b],
                    out_done=done[t, b],
                    out_terminated=terminated[t, b],
                    out_truncated=truncated[t, b],
                    out_info=env_info[t, b],
                )

            for b, env in envs_to_step:
                env.await_step()

            # calculate validity of samples in next time step
            # this might be required by the obs_transform
            valid[t + 1] = np.logical_and(valid[t], np.logical_not(done[t]))

        if self.get_bootstrap_value:
            # get bootstrap value for last observation in trajectory
            # if environment is already done, this value is invalid, but then
            # it will be ignored anyway
            sample_buffer["bootstrap_value"][...] = self.agent.value(
                self.sample_buffer["observation"][self.batch_spec.T]
            )

        # reset any environments that need reset in parallel
        envs_need_reset = [
            (b, env) for b, env in enumerate(self.envs) if env.needs_reset
        ]

        for b, env in envs_need_reset:
            # overwrite next first observation with reset observation
            env.reset_async(
                out_obs=observation[self.batch_spec.T, b],
                out_info=env_info[self.batch_spec.T - 1, b],
            )

        self.agent.reset_one([b for b, env in envs_need_reset])

        for b, env in envs_need_reset:
            env.await_step()

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj for env in self.envs for traj in env.collect_completed_trajs()
        ]

        # apply user-defined transforms
        if self.batch_transform is not None:
            sample_buffer = self.batch_transform(sample_buffer)

        return sample_buffer, completed_trajectories
