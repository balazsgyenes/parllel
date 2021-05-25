from functools import reduce
from typing import List, Tuple

from parllel.types.traj_info import TrajInfo
from .collections import Samples

class ClassicSampler:
    def __init__(self,
        batch_T: int,
        batch_B: int,
        get_bootstrap_value: bool = False,
        break_if_all_done: bool = False,
    ) -> None:
        self.batch_T = batch_T
        self.batch_B = batch_B
        self.get_bootstrap_value = get_bootstrap_value
        self.break_if_all_done = break_if_all_done

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
                env.step_async(action[t, b])

            for b, env in enumerate(self.envs):
                next_obs, reward[t, b], next_done, env_info[t, b] = env.await_step()
                if next_done:
                    next_obs = env.reset()
                    self.agent.reset_one(idx=b) # TODO: replace line
                observation[t+1, b] = next_obs
                done[t, b] = next_done

            if self.break_if_all_done and all(done[t]):
                # all done
                break
        
        #TODO: must ensure that observations for next batch are written to T+1,
        # even if trajectory is finished early

        """
        IDEA: create wait reset sampler with break_if_all_done functionality
        done property is stored by cage and used by sampler to filter, preventing calls to done cages
        reset obs is explictly requested from each cage at the end of the batch
        """

        # get bootstrap value if requested
        if self.get_bootstrap_value:
            # TODO: replace with agent.step()? Sampler chooses if rnn_state is advanced
            self.batch_buffer.agent.bootstrap_value[:] = self.agent.value(
                observation[t+1], action[t], reward[t])

        # collect all completed trajectories from envs
        completed_trajectories = reduce(
            lambda l, env: l.extend(env.collect_completed_trajs()),
            self.envs)

        return self.batch_buffer, completed_trajectories

    def close(self):
        pass