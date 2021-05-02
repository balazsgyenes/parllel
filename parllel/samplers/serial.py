from typing import List, Tuple
from functools import reduce

from parllel.types.traj_info import TrajInfo
from .types import Samples

class Sampler:
    def __init__(self,
        batch_T: int,
        batch_B: int,
        get_bootstrap_value: bool = False,
    ) -> None:
        self.batch_T = batch_T
        self.batch_B = batch_B
        self.get_bootstrap_value = get_bootstrap_value

        self.agent = None
        self.envs = None
        self.batch_buffer = None

    def initialize(self, agent, envs, batch_buffer: Samples) -> None:
        self.agent = agent
        self.envs = envs
        assert len(envs) == self.batch_B

        self.batch_buffer = batch_buffer

    def collect_batch(self, elapsed_steps) -> Tuple[Samples, List[TrajInfo]]:
        # get references to buffer elements
        action, agent_info, rnn_state = (
            self.batch_buffer.agent.action,
            self.batch_buffer.agent.agent_info,
            self.batch_buffer.agent.rnn_state,
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
        rnn_state.rotate()

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)
        
        # main sampling loop
        for t in range(0, self.batch_T):
            # agent observes environment and outputs actions
            action[t], agent_info[t], rnn_state[t+1] = self.agent.step(
                observation[t], action[t-1], reward[t-1], rnn_state[t])
            
            for b, env in enumerate(self.envs):
                env.step(action[t, b])

            for b, env in enumerate(self.envs):
                next_obs, reward[t, b], next_done, env_info[t, b] = env.await_step()
                if next_done:
                    next_obs = env.reset()
                    self.agent.reset_one(idx=b)
                observation[t+1, b] = next_obs
                done[t, b] = next_done
        
        # get bootstrap value if requested
        if self.get_bootstrap_value:
            # TODO: replace with agent.step()? Sampler chooses if rnn_state is advanced
            self.batch_buffer.agent.bootstrap_value[:] = self.agent.value(
                observation[t+1], action[t], reward[t])

        # collect all completed trajectories from envs
        completed_trajectories = reduce(
            function = lambda l, env: l.extend(env.collect_completed_trajs()),
            iterable = self.envs,
            initializer = [])

        return self.batch_buffer, completed_trajectories

    def shutdown(self):
        pass