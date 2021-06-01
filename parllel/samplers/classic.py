from functools import reduce
from typing import List, Tuple

from parllel.cages.cage import Cage
from parllel.types.traj_info import TrajInfo
from .collections import Samples

class ClassicSampler:
    """
    TODO: prev_action and prev_reward should be zeroes for start of trajectory
    This probably requires some additional local variables for temporary storage
    TODO: prevent calls to agent.step for environments that are done and waiting to be reset
    TODO: request reset obs from cage at the end of batch
    """
    def __init__(self,
        batch_T: int,
        batch_B: int,
        get_bootstrap_value: bool = False,
        wait_before_reset: bool = False,
    ) -> None:
        self.batch_T = batch_T
        self.batch_B = batch_B
        self.get_bootstrap_value = get_bootstrap_value
        self.wait_before_reset = wait_before_reset

    def initialize(self, agent, envs: List[Cage], batch_buffer: Samples) -> None:
        self.agent = agent
        self.envs = envs
        assert len(envs) == self.batch_B

        self.batch_buffer = batch_buffer

    def collect_batch(self, elapsed_steps) -> Tuple[Samples, List[TrajInfo]]:
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
        action.rotate()
        reward.rotate()

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)
        
        # main sampling loop
        for t in range(0, self.batch_T):
            # agent observes environment and outputs actions
            self.agent.step(observation[t], action[t-1], reward[t-1],
                env_ids=slice(None), out_action=action[t], out_agent_info=agent_info[t])

            for b, env in enumerate(self.envs):
                env.step_async(action[t, b],
                    out_obs=observation[t+1, b], out_reward=reward[t, b],
                    out_done=done[t, b], out_info=env_info[t, b])

            for b, env in enumerate(self.envs):
                env.await_step()
                if not self.wait_before_reset and done[t, b]:
                    self.agent.reset_one(env_index=b)

            # if self.wait_before_reset and all(done[t]):
            #     # all done
            #     break
        
        if self.wait_before_reset:
            for b, env in enumerate(self.envs):
                if done[self.batch_T, b]:
                    self.agent.reset_one(env_index=b)
                    env.collect_reset_obs(out_obs=observation[self.batch_T, b])

        # get bootstrap value if requested
        if self.get_bootstrap_value:
            self.batch_buffer.agent.bootstrap_value[:] = self.agent.value(
                observation[t+1], action[t], reward[t])

        # collect all completed trajectories from envs
        completed_trajectories = reduce(
            lambda l, env: l.extend(env.collect_completed_trajs()),
            self.envs)

        return self.batch_buffer, completed_trajectories

    def close(self):
        pass