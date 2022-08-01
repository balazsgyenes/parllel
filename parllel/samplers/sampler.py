from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy import random

from parllel.buffers import Samples
from parllel.cages import Cage, TrajInfo
from parllel.handlers import Handler
from parllel.types import BatchSpec


class Sampler(ABC):
    """Generates a batch of samples, where environments that are done are reset
    immediately. Use this sampler for non-recurrent agents.
    """
    def __init__(self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        agent: Handler,
        sample_buffer: Samples,
        max_steps_decorrelate: Optional[int] = None,
    ) -> None:
        self.batch_spec = batch_spec
        
        assert len(envs) == self.batch_spec.B
        self.envs = tuple(envs)
        
        self.agent = agent
        
        try:
            # try writing beyond the apparent bounds of the observation buffer
            observation = sample_buffer.env.observation
            observation[observation.last + 1] = 0
        except IndexError:
            raise TypeError("sample_buffer.env.observation must be a "
                "RotatingArray")
        self.sample_buffer = sample_buffer

        if max_steps_decorrelate is None:
            max_steps_decorrelate = 0
        self.max_steps_decorrelate = max_steps_decorrelate

        self.seed(seed=None)  # TODO: replace with seeding module

    def reset(self) -> None:
        """Prepare environments, agents and buffers for sampling.
        """
        self.reset_envs()
        if self.max_steps_decorrelate > 0:
            self.decorrelate_environments()
        self.reset_agent()

    def reset_envs(self) -> None:
        """Reset all environments. Reset observations are written to the end+1
        of the observation buffer, assuming that batch collection begins by
        rotating the batch buffer.
        """
        observation = self.sample_buffer.env.observation
        for b, env in enumerate(self.envs):
            # save reset observation to the end of buffer, since it will be 
            # rotated to the beginning
            env.reset_async(out_obs=observation[observation.last + 1, b])

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
        action = self.sample_buffer.agent.action
        observation, reward, done, env_info = (
            self.sample_buffer.env.observation,
            self.sample_buffer.env.reward,
            self.sample_buffer.env.done,
            self.sample_buffer.env.env_info,
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
                filter(lambda b_env: t < n_random_steps[b_env[0]],
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
