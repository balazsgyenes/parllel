from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy import random

from parllel.cages import Cage
from parllel.handlers import Handler
from parllel.types import BatchSpec, TrajInfo

from .collections import Samples


class Sampler(ABC):
    """Generates a batch of samples, where environments that are done are reset
    immediately. Use this sampler for non-recurrent agents.
    """
    def __init__(self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        agent: Handler,
        batch_buffer: Samples,
        max_steps_decorrelate: Optional[int] = None,
    ) -> None:
        self.batch_spec = batch_spec
        
        assert len(envs) == self.batch_spec.B
        self.envs = tuple(envs)
        
        self.agent = agent
        
        try:
            # try writing beyond the apparent bounds of the observation buffer
            T_last = self.batch_spec.T - 1
            batch_buffer.env.observation[T_last + 1] = 0
        except IndexError:
            raise TypeError("batch_samples.env.observation must be a "
                "RotatingArray")
        self.batch_buffer = batch_buffer

        if max_steps_decorrelate is None:
            max_steps_decorrelate = 0
        self.max_steps_decorrelate = max_steps_decorrelate

    def initialize(self) -> None:
        """Prepare environments, agents and buffers for sampling.
        """
        self.reset_envs()
        self.reset_agent()
        self.seed(seed=None)  # TODO: replace with seeding module
        if self.max_steps_decorrelate > 0:
            self.decorrelate_environments()

    def reset_envs(self) -> None:
        """Reset all environments. Reset observations are written to the end+1
        of the observation buffer, assuming that batch collection begins by
        rotating the batch buffer.
        """
        observation = self.batch_buffer.env.observation
        T_last = self.batch_spec.T - 1
        for b, env in enumerate(self.envs):
            # save reset observation to the end of buffer, since it will be 
            # rotated to the beginning
            env.reset_async(out_obs=observation[T_last + 1, b])

        # wait for envs to finish reset
        for b, env in enumerate(self.envs):
            env.await_step()

    def reset_agent(self) -> None:
        """Reset RNN state of agent, if it has one"""
        self.agent.reset()

    def seed(self, seed) -> None:
        self.rng = random.default_rng(seed)

    def decorrelate_environments(self) -> None:
        """Randomly step environments so they are not all synced up."""
        # get references to buffer elements
        action = self.batch_buffer.agent.action
        observation, reward, done, env_info = (
            self.batch_buffer.env.observation,
            self.batch_buffer.env.reward,
            self.batch_buffer.env.done,
            self.batch_buffer.env.env_info,
        )
        T_last = self.batch_spec.T - 1

        # get random number of steps between 0 and max for each env
        n_random_steps = self.rng.integers(
            low=0,
            high=self.max_steps_decorrelate,
            size=len(self.envs),
            dtype=np.int32,
        )

        envs_to_decorrelate = tuple(enumerate(self.envs))
        for t in range(self.max_steps_decorrelate):

            # filter out any environments that don't need to be stepped anymore
            envs_to_decorrelate = tuple(
                filter(lambda i_env: t < n_random_steps[i_env[0]],
                       envs_to_decorrelate)
            )

            env: Cage  # type declaration
            for b, env in envs_to_decorrelate:
                # always write data to last time step in the batch buffer, so
                # that previous values of first batch are correct after
                # rotating
                env.random_step_async(
                    out_action=action[T_last, b],
                    out_obs=observation[T_last + 1, b],
                    out_reward=reward[T_last, b],
                    out_done=done[T_last, b],
                    out_info=env_info[T_last, b]
                )

            for b, env in envs_to_decorrelate:
                env.await_step()

            # need to reset environment if it doesn't reset automatically
            for b, env in envs_to_decorrelate:
                if done[T_last, b] and env.wait_before_reset:
                    env.reset_async(out_obs=observation[T_last + 1, b])
                    env.await_step()

        # ignore any completed trajectories. The incomplete ones will be
        # continued during batch collection
        [env.collect_completed_trajs() for env in self.envs]

    @abstractmethod
    def collect_batch(self, elapsed_steps: int) -> Tuple[Samples, List[TrajInfo]]:
        pass

    def close(self) -> None:
        pass
