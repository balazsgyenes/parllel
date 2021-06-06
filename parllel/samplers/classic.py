from functools import reduce
from typing import List, Sequence, Tuple

import numpy as np

from parllel.buffers import Buffer
from parllel.cages import Cage
from parllel.handlers import Handler
from parllel.types.traj_info import TrajInfo
from .collections import Samples

class ClassicSampler:
    """
    TODO: prevent calls to agent.step for environments that are done and waiting to be reset
    """
    def __init__(self,
        batch_T: int,
        batch_B: int,
        get_bootstrap_value: bool = False,
        reset_only_after_batch: bool = False,
    ) -> None:
        self.batch_T = batch_T
        self.batch_B = batch_B
        self.get_bootstrap_value = get_bootstrap_value
        self.reset_only_after_batch = reset_only_after_batch

    def initialize(self,
        agent: Handler,
        envs: Sequence[Cage],
        batch_buffer: Samples,
        step_action: Buffer,
        step_reward: Buffer,
    ) -> None:
        self.agent = agent
        self.envs = tuple(envs)
        assert len(envs) == self.batch_B
        self.batch_buffer = batch_buffer
        self.step_action = step_action
        self.step_reward = step_reward

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
        step_action = self.step_action
        step_reward = self.step_reward

        # rotate last values from previous batch to become previous values
        observation.rotate()
        # action.rotate()
        # reward.rotate()

        #TODO: ensure correct initial values for step_action and step_reward

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)
        
        # main sampling loop
        for t in range(0, self.batch_T):
            # agent observes environment and outputs actions
            # step_action and step_reward are from previous time step (t-1)
            self.agent.step(observation[t], step_action, step_reward,
                env_ids=slice(None), out_action=step_action, out_agent_info=agent_info[t])

            for b, env in enumerate(self.envs):
                env.step_async(step_action[b],
                    out_obs=observation[t+1, b], out_reward=step_reward[b],
                    out_done=done[t, b], out_info=env_info[t, b])

            # commit step_action to sample buffer, it was written by the agent
            action[t] = step_action

            for b, env in enumerate(self.envs):
                env.await_step()

            # commit step_reward to sample buffer, it was written by the environments
            reward[t] = step_reward

            if self.reset_only_after_batch:
                # no reset is required during batch
                if np.all(done[t]):
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

        batch_samples = np.asarray(self.batch_buffer[:(t+1)])

        return batch_samples, completed_trajectories

    def close(self):
        pass