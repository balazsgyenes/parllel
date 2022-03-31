from typing import List, Optional, Sequence, Tuple

import numpy as np

from parllel.buffers.utils import buffer_map, buffer_rotate
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
        """Generates samples for training recurrent agents.

        TODO: obs transform should not include data from invalid time steps,
        since this would distort statistics. When is valid calculated?
        """
        for cage in envs:
            if not cage.wait_before_reset:
                raise ValueError("RecurrentSampler expects cages that do not "
                    "reset environments until the end of a batch. Set "
                    "wait_before_reset=True")
        
        try:
            # try writing beyond the apparent bounds of the action buffer
            T_last = self.batch_spec.T - 1
            batch_buffer.agent.action[T_last + 1] = 0
        except IndexError:
            raise TypeError("batch_samples.agent.action must be a "
                "RotatingArray")
        
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

    def reset_agent(self) -> None:
        super().reset_agent()
        # zero out previous action of very first step (after rotation)
        # this matches internal state of agent, which also begins at zero
        action = self.batch_buffer.agent.action
        action[action.last] = 0

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
        envs = self.envs
        last_T = self.batch_spec.T - 1

        # rotate last values from previous batch to become previous values
        buffer_rotate(self.batch_buffer)

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)
        
        # main sampling loop
        b_not_done_yet = list(range(len(envs)))
        for t in range(self.batch_spec.T):

            # get a list of environments that are not done yet
            # we want to avoid stepping these
            b_not_done_yet = list(
                filter(lambda b: not envs[b].already_done, b_not_done_yet)
            )

            if not b_not_done_yet:
                # all done, we can stop sampling now
                break

            # apply any transforms to the observation before the agent steps
            self.batch_samples = self.obs_transform(self.batch_buffer,
                (t, b_not_done_yet))

            # agent observes environment and outputs actions
            self.agent.step(observation[t], out_action=action[t],
                out_agent_info=agent_info[t])

            for b in b_not_done_yet:
                envs[b].step_async(action[t, b],
                    out_obs=observation[t+1, b], out_reward=reward[b],
                    out_done=done[t, b], out_info=env_info[t, b])

            for b in b_not_done_yet:
                envs[b].await_step()

        if self.get_bootstrap_value:
            # get bootstrap value for last observation in trajectory
            # if environment is already done, this value is invalid, but then
            # it will be ignored anyway
            self.batch_buffer.agent.bootstrap_value[:] = self.agent.value(
                observation[last_T + 1])

        for b, env in enumerate(self.envs):
            if env.already_done:
                self.agent.reset_one(env_index=b)
                # overwrite next first observation with reset observation
                env.reset_async(out_obs=observation[last_T + 1, b])
                env.await_step()
                # previous action for next batch
                action[last_T + 1, b] = 0

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj for env in self.envs for traj
            in env.collect_completed_trajs()
            ]

        batch_samples = self.batch_transform(self.batch_buffer)

        # convert to underlying numpy array
        batch_samples = buffer_map(np.asarray, batch_samples)

        return batch_samples, completed_trajectories
