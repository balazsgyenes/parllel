from typing import List, Optional, Sequence, Tuple

import numpy as np

from parllel.buffers import Samples
from parllel.buffers.utils import buffer_rotate
from parllel.cages import Cage, TrajInfo
from parllel.handlers import Handler
from parllel.transforms import Transform
from parllel.types import BatchSpec

from .sampler import Sampler


class EvalSampler(Sampler):
    """Generates a batch of samples, where environments that are done are reset
    immediately. Use this sampler for non-recurrent agents.
    """
    def __init__(self,
        max_traj_length: int,
        min_trajectories: int,
        envs: Sequence[Cage],
        agent: Handler,
        step_buffer: Samples,
        obs_transform: Optional[Transform] = None,
    ) -> None:
        for cage in envs:
            if cage.wait_before_reset:
                raise ValueError("EvaluationSampler expects cages that reset"
                    " environments immediately. Set wait_before_reset=False")
        
        super().__init__(
            batch_spec = BatchSpec(max_traj_length, len(envs)),
            envs = envs,
            agent = agent,
            batch_buffer = step_buffer,
        )

        self.min_trajectories = min_trajectories

        if obs_transform is None:
            obs_transform = lambda x, t: x
        self.obs_transform = obs_transform

        # prepare cages for sampling
        self.reset_envs()

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

        # reset all environments
        self.reset_envs()

        # rotate last values from previous batch to become previous values
        buffer_rotate(self.batch_buffer)

        # prepare agent for sampling
        self.agent.eval_mode(elapsed_steps)
        self.agent.reset()
        
        n_completed_trajs = 0

        # main sampling loop
        for _ in range(self.batch_spec.T):

            # apply any transforms to the observation before the agent steps
            self.batch_samples = self.obs_transform(self.batch_buffer, 0)

            # agent observes environment and outputs actions
            self.agent.step(observation[0], out_action=action[0],
                out_agent_info=agent_info[0])

            for b, env in enumerate(self.envs):
                env.step_async(action[0, b],
                    out_obs=observation[0, b], out_reward=reward[0, b],
                    out_done=done[0, b], out_info=env_info[0, b])

            for b, env in enumerate(self.envs):
                env.await_step()

            # if environment is done, reset agent
            # environment has already been reset inside cage
            if np.any(dones := done[0]):
                n_completed_trajs += np.sum(dones)
                if n_completed_trajs >= self.min_trajectories:
                    break
                self.agent.reset_one(np.asarray(dones))
        
        # collect all completed trajectories from envs
        completed_trajectories = [
            traj
            for env in self.envs
            for traj in env.collect_completed_trajs()
        ]

        return completed_trajectories
