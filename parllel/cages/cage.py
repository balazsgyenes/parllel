from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Union

import gym
from gym.wrappers import TimeLimit as GymTimeLimit

from parllel.arrays import Array
from parllel.buffers import (Buffer, buffer_asarray, dict_to_namedtuple,
    namedtuple_to_dict)

from .collections import EnvStep, EnvSpaces
from .traj_info import TrajInfo


class Cage(ABC):
    """Cages abstract communication between the sampler and the environments.

    :param EnvClass (Callable): Environment class or factory function
    :param env_kwargs (Dict): Key word arguments that should be passed to the
        `__init__` of `EnvClass` or to the factory function
    :param TrajInfoClass (Callable): TrajectoryInfo class or factory function
    wait_before_reset (bool): If True, environment does not reset when done
        until `reset_async` is called, and `already_done` is set to True. If
        False (default), the environment resets immediately.
    """
    def __init__(self,
        EnvClass: Callable,
        env_kwargs: Dict,
        TrajInfoClass: Callable,
        wait_before_reset: bool = False,
    ) -> None:
        self.EnvClass = EnvClass
        self.env_kwargs = env_kwargs
        self.TrajInfoClass = TrajInfoClass
        self.wait_before_reset = wait_before_reset

        self._already_done: bool = False
        self._render: bool = False

    def _create_env(self) -> None:
        self._completed_trajs: List[TrajInfo] = []
        self._traj_info: TrajInfo = self.TrajInfoClass()

        self._env: gym.Env = self.EnvClass(**self.env_kwargs)
        self._env.reset()

        # determine if environment also wrapped with gym's TimeLimit
        env_unwrapped = self._env
        self._time_limit = isinstance(env_unwrapped, GymTimeLimit)
        while not self._time_limit and hasattr(env_unwrapped, "env"):
            env_unwrapped = env_unwrapped.env
            self._time_limit = isinstance(env_unwrapped, GymTimeLimit)

        # save obs and action spaces for easy access
        self._spaces = EnvSpaces(
            observation=self._env.observation_space,
            action=self._env.action_space,
        )

    def _close_env(self) -> None:
        self._env.close()

    @property
    def spaces(self) -> EnvSpaces:
        return self._spaces

    @property
    def already_done(self) -> bool:
        return self._already_done

    @property
    def render(self) -> bool:
        return self._render

    @render.setter
    def render(self, value: bool) -> None:
        self._render = value

    @abstractmethod
    def set_samples_buffer(self,
        action: Buffer,
        obs: Buffer,
        reward: Buffer,
        done: Array,
        info: Buffer,
    ) -> None:
        raise NotImplementedError

    def _step_env(self, action: Buffer) -> Tuple[Buffer, Buffer, Buffer, Buffer]:
        """If any out parameter is given, they must all be given. 
        """
        # if rendering, render before step is taken so that the renderings
        # line up with the corresponding observation
        if self._render:
            rendering = self._env.render(mode="rgb_array")
        
        # get underlying numpy arrays and convert to dict if needed
        action = buffer_asarray(action)
        action = namedtuple_to_dict(action)

        obs, reward, done, env_info = self._env.step(action)
        self._traj_info.step(obs, action, reward, done, env_info)
        
        if self._time_limit:
            env_info["timeout"] = env_info.pop("TimeLimit.truncated", False)

        if self._render:
            env_info["rendering"] = rendering

        return obs, reward, done, env_info

    def _random_step_env(self) -> Tuple[Buffer, Buffer, Buffer, Buffer, Buffer]:

        action = self._env.action_space.sample()
        action = dict_to_namedtuple(action, "action")

        obs, reward, done, env_info = self._step_env(action)

        if done:
            # reset immediately and overwrite last observation
            obs = self._reset_env()
        
        return action, obs, reward, done, env_info

    def _reset_env(self) -> Buffer:
        # store finished trajectory and start new one
        self._completed_trajs.append(self._traj_info)
        self._traj_info = self.TrajInfoClass()
        return self._env.reset()

    @abstractmethod
    def await_step(self) -> Union[EnvStep, Tuple[Buffer, ...], Buffer, bool]:
        """Wait for the asynchronous step to finish and return the results.
        If step_async, reset_async, or random_step_async was called previously
        with input arguments, returns whether the environment is now done and
        needs reset.
        If step_async was called previously without output arguments, returns
        the EnvStep.
        If reset_async was called previously without output arguments, returns
        the reset observation.
        If random_step_async was called previously without output arguments,
        returns the action, observation, reward, done and env_info as a tuple.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_completed_trajs(self) -> List[TrajInfo]:
        raise NotImplementedError

    @abstractmethod
    def random_step_async(self, *,
        out_action: Buffer = None,
        out_obs: Buffer = None,
        out_reward: Buffer = None,
        out_done: Buffer = None,
        out_info: Buffer = None
    ) -> None:
        """Take a step with a random action from the env's action space. The
        env resets automatically if done, regardless of the value of
        wait_before_reset.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_async(self, *, out_obs: Buffer = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
