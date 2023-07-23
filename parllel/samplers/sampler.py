from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy import random

import parllel.logger as logger
from parllel import Array, ArrayDict
from parllel.agents import Agent
from parllel.cages import Cage, TrajInfo
from parllel.types import BatchSpec


class Sampler(ABC):
    def __init__(
        self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        agent: Agent,
        sample_buffer: ArrayDict[Array],
        max_steps_decorrelate: int | None = None,
    ) -> None:
        self.batch_spec = batch_spec

        assert len(envs) == self.batch_spec.B
        self.envs = tuple(envs)

        self.agent = agent

        try:
            # try writing beyond the expected bounds of the observation buffer
            sample_buffer["observation"][self.batch_spec.T] = 0
        except IndexError:
            raise ValueError("sample_buffer[`observation`] must have padding >= 1")
        self.sample_buffer = sample_buffer

        if max_steps_decorrelate is None:
            max_steps_decorrelate = 0
        self.max_steps_decorrelate = max_steps_decorrelate

        self.seed(seed=None)  # TODO: replace with seeding module

    def reset(self) -> None:
        """Prepare environments, agents and buffers for sampling."""
        self.reset_envs()
        if self.max_steps_decorrelate > 0:
            self.decorrelate_environments()
        self.reset_agent()

    def reset_envs(self) -> None:
        """Reset all environments. Reset observations are written to the end+1
        of the observation buffer, assuming that batch collection begins by
        rotating the batch buffer.
        """
        logger.debug(f"{type(self).__name__}: Resetting sampler buffer state.")
        self.sample_buffer.reset()
        logger.debug(f"{type(self).__name__}: Resetting all environments.")
        observation = self.sample_buffer["observation"]
        env_info = self.sample_buffer["env_info"]
        for b, env in enumerate(self.envs):
            # save reset observation to the end of buffer, since it will be
            # rotated to the beginning
            env.reset_async(
                out_obs=observation[self.batch_spec.T, b],
                out_info=env_info[self.batch_spec.T, b],
            )

        # wait for envs to finish reset
        for b, env in enumerate(self.envs):
            env.await_step()

        # discard the trajectories that were just forcefully completed
        [env.collect_completed_trajs() for env in self.envs]

    def reset_agent(self) -> None:
        """Reset RNN state of agent, if it has one"""
        logger.debug(f"{type(self).__name__}: Resetting agent.")
        self.agent.reset()

    def seed(self, seed) -> None:
        self.rng = random.default_rng(seed)

    def decorrelate_environments(self) -> None:
        """Randomly step environments so they are not all synced up."""
        logger.info(
            f"{type(self).__name__}: Decorrelating environments with up to "
            f"{self.max_steps_decorrelate} random steps each."
        )
        # get references to buffer elements
        action = self.sample_buffer["action"]
        observation = self.sample_buffer["observation"]
        reward = self.sample_buffer["reward"]
        done = self.sample_buffer["done"]
        terminated = self.sample_buffer["terminated"]
        truncated = self.sample_buffer["truncated"]
        env_info = self.sample_buffer["env_info"]
        T_last = self.batch_spec.T - 1

        # get random number of steps between 0 and max for each env
        n_random_steps = self.rng.integers(
            low=0,
            high=self.max_steps_decorrelate,
            size=len(self.envs),
            dtype=np.int32,
        )

        env_to_step = list(enumerate(self.envs))
        for t in range(self.max_steps_decorrelate):
            # filter out any environments that don't need to be stepped anymore
            env_to_step = [(b, env) for b, env in env_to_step if t <= n_random_steps[b]]

            if not env_to_step:
                # all done, we can stop decorrelating now
                break

            for b, env in env_to_step:
                # always write data to last time step in the batch buffer, so
                # that previous values of first batch are correct after
                # rotating
                env.random_step_async(
                    out_action=action[T_last, b],
                    out_obs=observation[T_last + 1, b],
                    out_reward=reward[T_last, b],
                    out_done=done[T_last, b],
                    out_terminated=terminated[T_last, b],
                    out_truncated=truncated[T_last, b],
                    out_info=env_info[T_last, b],
                )

            for b, env in env_to_step:
                env.await_step()

            # no need to reset environments, since they are always reset
            # automatically when calling random_step_async

        # discard any completed trajectories. The incomplete ones will be
        # continued during batch collection
        [env.collect_completed_trajs() for env in self.envs]

    @abstractmethod
    def collect_batch(
        self,
        elapsed_steps: int,
    ) -> tuple[ArrayDict[Array], list[TrajInfo]]:
        pass

    def close(self) -> None:
        pass
