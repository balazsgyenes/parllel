from functools import reduce
from typing import List, Optional, Sequence, Tuple

import numpy as np

from parllel.arrays import buffer_from_example
from parllel.buffers.utils import buffer_map, buffer_method, buffer_rotate
from parllel.cages import Cage
from parllel.handlers import Handler
from parllel.transforms import Transform
from parllel.types import BatchSpec, TrajInfo

from .collections import Samples
from .sampler import Sampler


class RecurrentSampler(Sampler):
    def __init__(self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        agent: Handler,
        batch_buffer: Samples,
        max_steps_decorrelate: Optional[int] = None,
        get_bootstrap_value: bool = False,
        obs_transform: Transform = None,
        batch_transform: Transform = None,
    ) -> None:
        for cage in envs:
            if not cage.wait_before_reset:
                raise ValueError("RecurrentSampler expects cages that do not "
                    "reset environments until the end of a batch. Set "
                    "wait_before_reset=True")
        
        super().__init__(
            batch_spec = batch_spec,
            envs = envs,
            agent = agent,
            batch_buffer = batch_buffer,
            max_steps_decorrelate = max_steps_decorrelate,
        )

        self.get_bootstrap_value = get_bootstrap_value
        
        if obs_transform is None:
            obs_transform = lambda x, t: x
        self.obs_transform = obs_transform

        if batch_transform is None:
            batch_transform = lambda x: x
        self.batch_transform = batch_transform

        # prepare cages for sampling
        self.initialize()

    def initialize(self) -> None:
        super().initialize()

        # create array to hold "previous action" temporarily
        self._step_action = buffer_from_example(self.batch_buffer.agent.action[0], ())

    def collect_batch(self, elapsed_steps: int) -> Tuple[Samples, List[TrajInfo]]:
        # get references to buffer elements
        action, agent_info = (
            self.batch_buffer.agent.action,
            self.batch_buffer.agent.agent_info,
        )
        observation, reward, done, env_info = (
            self.batch_buffer.env.observation,
            self.batch_buffer.env.reward,
            self.batch_buffer.env.done,
            self.batch_buffer.env.env_info,
        )
        step_action = self._step_action
        
        # initialize step_action to the final value from the last batch
        last_T = self.batch_spec.T - 1
        step_action[...] = action[last_T]

        # rotate last values from previous batch to become previous values
        buffer_rotate(self.batch_buffer)

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)
        
        # main sampling loop
        envs_to_step = tuple(enumerate(self.envs))
        for t in range(self.batch_spec.T):

            envs_to_step = tuple(
                filter(lambda b_env: not b_env[1].already_done, envs_to_step)
            )

            # TODO:
            # filter envs sent to obs_transform so that e.g. statistics are
            # not affected by environments that are done
            # probably can make envs_to_step cleaner by maintaining an np array
            # of indices to step

            # apply any transforms to the observation before the agent steps
            self.batch_samples = self.obs_transform(self.batch_buffer, t)

            # agent observes environment and outputs actions
            # step_action and step_reward are from previous time step (t-1)
            self.agent.step(observation[t], step_action, out_action=step_action,
                out_agent_info=agent_info[t])

            for b, env in enumerate(self.envs):
                env.step_async(step_action[b],
                    out_obs=observation[t+1, b], out_reward=reward[b],
                    out_done=done[t, b], out_info=env_info[t, b])

            # commit step_action to sample buffer, it was written by the agent
            action[t] = step_action

            for b, env in enumerate(self.envs):
                env.await_step()

            if self.reset_only_after_batch:
                # no reset is required during batch
                if all(env.already_done for env in self.envs):
                    # all environments done, stop sampling early
                    # copy last observations to the end of batch buffer
                    # after the array is rotated, this will become observation[0]
                    observation[self.batch_T] = observation[t+1]
                    break
            else:
                # after step_reward is committed, reset agent, previous action and
                # previous reward
                for b, env in enumerate(self.envs):
                    if done[t, b]:
                        self.agent.reset_one(env_index=b)
                        # previous action for next step
                        step_action[b] = 0
                        # previous reward for next step
                        step_reward[b] = 0

        # t is now the index of the last time step in batch
        # t <= (batch_T - 1)

        if self.get_bootstrap_value:
            # get bootstrap value for last observation in trajectory
            # if environment is already done, this value is invalid, but then
            # it will be ignored anyway
            self.batch_buffer.agent.bootstrap_value[:] = self.agent.value(
                observation[t+1], action[t], reward[t])

        if self.reset_only_after_batch:
            for b, env in enumerate(self.envs):
                if done[t, b]:
                    self.agent.reset_one(env_index=b)
                    # overwrite next previous observation with reset observation
                    env.collect_reset_obs(out_obs=observation[self.batch_T, b])
                    # previous action for next batch
                    step_action[b] = 0
                    # previous reward for next batch
                    step_reward[b] = 0

        # collect all completed trajectories from envs
        completed_trajectories = [traj for env in self.envs for traj in env.collect_completed_trajs()]

        batch_samples = buffer_map(np.asarray, self.batch_buffer[:(t+1)])

        return batch_samples, completed_trajectories

    def close(self):
        buffer_method(self._step_action, "close")
        buffer_method(self._step_action, "destroy")
