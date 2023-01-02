from typing import List, Optional, Sequence, Tuple

import numpy as np

from parllel.buffers import Samples
from parllel.buffers.utils import buffer_rotate
from parllel.cages import Cage, TrajInfo
from parllel.handlers import Handler
from parllel.transforms import Transform
from parllel.types import BatchSpec

from .sampler import Sampler


class RecurrentSampler(Sampler):
    def __init__(self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        agent: Handler,
        sample_buffer: Samples,
        max_steps_decorrelate: Optional[int] = None,
        get_bootstrap_value: bool = False,
        obs_transform: Optional[Transform] = None,
        batch_transform: Optional[Transform] = None,
    ) -> None:
        """Generates samples for training recurrent agents.
        """
        for cage in envs:
            if cage.reset_automatically:
                raise ValueError(
                    "RecurrentSampler expects cages that do not reset "
                    "environments until commanded to. Set "
                    "`reset_automatically=False`."
                )

        # verify that initial_rnn_state field exists
        if not hasattr(sample_buffer.agent, "initial_rnn_state"):
            raise ValueError("RecurrentSampler expects a buffer field at "
                "sample_buffer.agent.initial_rnn_state. Please allocate this.")
        
        # verify that valid field exists
        if not hasattr(sample_buffer.env, "valid"):
            raise ValueError("RecurrentSampler expects a buffer field at "
                "sample_buffer.env.valid. Please allocate this.")

        # verify that valid is a RotatingArray
        try:
            # try writing beyond the apparent bounds of the action buffer
            sample_buffer.env.valid[batch_spec.T] = 0
        except IndexError:
            raise TypeError("sample_buffer.env.valid must be a "
                "RotatingArray")

        super().__init__(
            batch_spec=batch_spec,
            envs=envs,
            agent=agent,
            sample_buffer=sample_buffer,
            max_steps_decorrelate=max_steps_decorrelate,
        )

        if get_bootstrap_value and not hasattr(self.sample_buffer.agent,
                "bootstrap_value"):
            raise ValueError("Requested bootstrap value from agent, but "
                "sample_buffer.agent.bootstrap_value does not exist. Please "
                "allocate it.")
        self.get_bootstrap_value = get_bootstrap_value
        
        self.obs_transform = obs_transform
        self.batch_transform = batch_transform

        # prepare cages for sampling
        self.reset()

    def collect_batch(self, elapsed_steps: int) -> Tuple[Samples, List[TrajInfo]]:
        # get references to buffer elements
        action, agent_info = (
            self.sample_buffer.agent.action,
            self.sample_buffer.agent.agent_info,
        )
        observation, reward, done, env_info, valid = (
            self.sample_buffer.env.observation,
            self.sample_buffer.env.reward,
            self.sample_buffer.env.done,
            self.sample_buffer.env.env_info,
            self.sample_buffer.env.valid,
        )
        sample_buffer = self.sample_buffer

        # rotate last values from previous batch to become previous values
        buffer_rotate(sample_buffer)

        # prepare agent for sampling
        self.agent.sample_mode(elapsed_steps)
        
        self.agent.initial_rnn_state(
            out_rnn_state=sample_buffer.agent.initial_rnn_state
        )
        
        # first time step is always valid, rest are invalid by default
        valid[0] = True
        valid[1:] = False

        # main sampling loop
        envs_to_step = list(enumerate(self.envs))
        for t in range(self.batch_spec.T):

            # get a list of environments that are not done yet
            # we want to avoid stepping these
            envs_to_step = [(b, env) for b, env in envs_to_step if not env.needs_reset]

            if not envs_to_step:
                # all done, we can stop sampling now
                break

            # apply any transforms to the observation before the agent steps
            if self.obs_transform is not None:
                self.batch_samples = self.obs_transform(sample_buffer, t)

            # agent observes environment and outputs actions
            self.agent.step(observation[t], out_action=action[t],
                out_agent_info=agent_info[t])

            for b, env in envs_to_step:
                env.step_async(action[t, b],
                    out_obs=observation[t+1, b], out_reward=reward[t, b],
                    out_done=done[t, b], out_info=env_info[t, b])

            for b, env in envs_to_step:
                env.await_step()

            # calculate validity of samples in next time step
            # this might be required by the obs_transform
            valid[t + 1] = np.logical_and(valid[t], np.logical_not(done[t]))
        
        if self.get_bootstrap_value:
            # get bootstrap value for last observation in trajectory
            # if environment is already done, this value is invalid, but then
            # it will be ignored anyway
            self.agent.value(
                observation[self.batch_spec.T],
                out_value=sample_buffer.agent.bootstrap_value,
            )

        # reset any environments that need reset in parallel
        envs_need_reset = [(b, env) for b, env in enumerate(self.envs) if env.needs_reset]
        
        for b, env in envs_need_reset:
            # overwrite next first observation with reset observation
            env.reset_async(out_obs=observation[self.batch_spec.T, b])
            
        self.agent.reset_one([b for b, env in envs_need_reset])

        for b, env in envs_need_reset:
            env.await_step()

        # collect all completed trajectories from envs
        completed_trajectories = [
            traj
            for env in self.envs
            for traj in env.collect_completed_trajs()
        ]

        # apply user-defined transforms
        if self.batch_transform is not None:
            sample_buffer = self.batch_transform(sample_buffer)

        return sample_buffer, completed_trajectories
