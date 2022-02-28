from typing import List, Sequence, Tuple

import numpy as np

from parllel.buffers import buffer_func
from parllel.cages import Cage
from parllel.handlers import Handler
from parllel.types.traj_info import TrajInfo
from .collections import Samples

class MiniSampler:
    """Generates a batch of samples, where environments that are done are reset
    immediately. Use this sampler for non-recurrent agents.
    """
    def __init__(self,
        batch_T: int,
        batch_B: int,
        get_bootstrap_value: bool = False,
    ) -> None:
        self.batch_T = batch_T
        self.batch_B = batch_B
        self.get_bootstrap_value = get_bootstrap_value

    def initialize(self,
        agent: Handler,
        envs: Sequence[Cage],
        batch_buffer: Samples,
    ) -> Tuple[Samples, List[TrajInfo]]:
        self.agent = agent
        self.envs = tuple(envs)
        assert len(envs) == self.batch_B
        self.batch_buffer = batch_buffer

        # t = batch_T + 1 will become the observation for the first action
        # ensure that it is properly zeroed
        self.batch_buffer.env.observation[self.batch_T] = 0
        
        # get example of a batch of samples
        # TODO: is this necessary? could just return random Samples buffer
        example_batch = self.collect_batch(0)

        # randomly step environments so they are not all synced up
        self.decorrelate_environments()

        return example_batch

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

        # rotate last values from previous batch to become previous values
        observation.rotate()

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)
        
        # main sampling loop
        for t in range(0, self.batch_T):
            # agent observes environment and outputs actions
            # step_action and step_reward are from previous time step (t-1)
            self.agent.step(observation[t], None, out_action=action[t],
                out_agent_info=agent_info[t])

            for b, env in enumerate(self.envs):
                env.step_async(action[t, b],
                    out_obs=observation[t+1, b], out_reward=reward[t, b],
                    out_done=done[t, b], out_info=env_info[t, b])

            for b, env in enumerate(self.envs):
                env.await_step()

            # if environment is done, reset agent, previous action and
            # previous reward
            for b, env in enumerate(self.envs):
                if done[t, b]:
                    self.agent.reset_one(env_index=b)
                    env.reset(out_obs=observation[t+1, b])

        # t is now the index of the last time step in batch
        # t <= (batch_T - 1)

        if self.get_bootstrap_value:
            # get bootstrap value for last observation in trajectory
            self.batch_buffer.agent.bootstrap_value[:] = self.agent.value(
                observation[t+1], None)

        # collect all completed trajectories from envs
        completed_trajectories = [traj for env in self.envs for traj in env.collect_completed_trajs()]

        batch_samples = buffer_func(np.asarray, self.batch_buffer[:(t+1)])

        return batch_samples, completed_trajectories

    def decorrelate_environments(self) -> None:
        # TODO: model this off of sampling loop
        pass

    def close(self):
        pass
