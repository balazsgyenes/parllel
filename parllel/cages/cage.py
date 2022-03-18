from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import gym
from gym.wrappers import TimeLimit as GymTimeLimit

from parllel.arrays import Array
from parllel.buffers import Buffer, buffer_func
from parllel.buffers.named_tuple import namedtuple_to_dict
from parllel.types.traj_info import TrajInfo

from .collections import EnvStep, EnvSpaces


INVALID_STEP_RESULT: str = "This is an invalid step result"


class Cage:
    """Cages abstract communication between the sampler and the environments.

    Note: a sentinel object is used in place of None where an error
    should be triggered. None is ignored by NamedArrayTuple, causing operations
    to silently fail where we want an error to be thrown.

    Args:
        EnvClass (Callable): TODO
        env_kwargs (Dict): Key word arguments that should be passed to the
            `__init__` of `EnvClass`
        TrajInfoClass (Callable): TODO
        traj_info_kwargs (Dict): Key word arguments that should be passed to
            the `__init__` of `TrajInfoClass`
        wait_before_reset (bool): TODO

    TODO: merge collect_deferred_reset and reset_async. The fact that the reset
    has already been done in one of these cases is an internal implementation
    detail
    """
    def __init__(self,
        EnvClass: Callable,
        env_kwargs: Dict,
        TrajInfoClass: Callable,
        traj_info_kwargs: Dict,
        wait_before_reset: bool = False,
        samples_buffer: Optional[Sequence[Buffer]] = None,
    ) -> None:
        self.EnvClass = EnvClass
        self.env_kwargs = env_kwargs
        self.TrajInfoClass = TrajInfoClass
        self.traj_info_kwargs = traj_info_kwargs
        self.wait_before_reset = wait_before_reset

        self._already_done: bool = False
        self._create_env()

        if samples_buffer is not None:
            self.set_samples_buffer(*samples_buffer)

    def _create_env(self) -> None:
        self._completed_trajs: List[TrajInfo] = []
        self._traj_info: TrajInfo = self.TrajInfoClass(**self.traj_info_kwargs)
        self._step_result: Union[EnvStep, str] = INVALID_STEP_RESULT
        self._reset_obs: Union[Buffer, str] = INVALID_STEP_RESULT

        self._env: gym.Env = self.EnvClass(**self.env_kwargs)
        self._env.reset()

        # determine if environment also wrapped with gym's TimeLimit
        env_unwrapped = self._env
        self._time_limit = isinstance(env_unwrapped, GymTimeLimit)
        while not self._time_limit and hasattr(env_unwrapped, "env"):
            env_unwrapped = env_unwrapped.env
            self._time_limit = isinstance(env_unwrapped, GymTimeLimit)

        self._spaces = EnvSpaces(
            observation=self._env.observation_space,
            action=self._env.action_space,
        )

    def get_example_output(self) -> EnvStep:
        # get example of env step output
        self._env.reset()
        sample_action = self._env.action_space.sample()
        obs, reward, done, info = self._env.step(sample_action)

        if self._time_limit:
            # we will be using this example to allocate a buffer, so all fields
            # that could be present need to be present
            info["timeout"] = info.pop("TimeLimit.truncated", False)

        return EnvStep(obs, reward, done, info)

    @property
    def spaces(self) -> EnvSpaces:
        return self._spaces

    @property
    def already_done(self) -> bool:
        return self._already_done

    def set_samples_buffer(self, action: Buffer, obs: Buffer, reward: Buffer,
                           done: Array, info: Buffer) -> None:
        pass

    def step_async(self,
        action: Buffer, *,
        out_obs: Buffer = None,
        out_reward: Buffer = None,
        out_done: Buffer = None,
        out_info: Buffer = None,
    ) -> None:
        """If any out parameter is given, they must all be given. 
        """
        # get underlying numpy arrays and convert to dict if needed
        action = buffer_func(np.asarray, action)
        action = namedtuple_to_dict(action)

        obs, reward, done, env_info = self._env.step(action)
        self._traj_info.step(obs, action, reward, done, env_info)
        
        if self._time_limit:
            env_info["timeout"] = env_info.pop("TimeLimit.truncated", False)

        if done:
            # store finished trajectory and start new one
            self._completed_trajs.append(self._traj_info)
            self._traj_info = self.TrajInfoClass(**self.traj_info_kwargs)
            if self.wait_before_reset:
                # start environment reset asynchronously
                self._defer_env_reset()
                # store done state
                self._already_done = True
            else:
                # reset immediately and overwrite last observation
                obs = self._env.reset()
    
        if any(out is None for out in (out_obs, out_reward, out_done, out_info)):
            self._step_result = EnvStep(obs, reward, done, env_info)
        else:
            out_obs[:] = obs
            out_reward[:] = reward
            out_done[:] = done
            out_info[:] = env_info
            self._step_result = INVALID_STEP_RESULT

    def _defer_env_reset(self) -> None:
        self._reset_obs = self._env.reset()

    def await_step(self) -> Union[EnvStep, Tuple[Buffer, EnvStep], Buffer]:
        result = self._step_result
        self._step_result = INVALID_STEP_RESULT
        return result

    def collect_deferred_reset(self, *, out_obs: Buffer = None) -> Optional[Buffer]:
        result = self._reset_obs
        self._already_done = False
        self._reset_obs = INVALID_STEP_RESULT
        if out_obs is not None:
            out_obs[:] = result
        else:
            return result

    def collect_completed_trajs(self) -> List[TrajInfo]:
        completed_trajs = self._completed_trajs
        self._completed_trajs = []
        return completed_trajs

    def random_step_async(self, *,
        out_action: Buffer = None,
        out_obs: Buffer = None,
        out_reward: Buffer = None,
        out_done: Buffer = None,
        out_info: Buffer = None
    ) -> None:
        """Take a step with a random action from the env's action space.
        """
        action = self.spaces.action.sample()
        wait_before_reset = self.wait_before_reset
        self.wait_before_reset = False
        self.step_async(action, out_obs, out_reward, out_done, out_info)
        self.wait_before_reset = wait_before_reset

        if self._step_result is not INVALID_STEP_RESULT:
            self._step_result = (action, self._step_result)
        else:
            out_action[:] = action

    def reset_async(self, out_obs: Buffer = None) -> None:
        _reset_obs = self._env.reset()
        self._traj_info = self.TrajInfoClass(**self.traj_info_kwargs)

        if out_obs is None:
            self._step_result = _reset_obs
        else:
            out_obs[:] = _reset_obs

    def close(self) -> None:
        self._env.close()
