from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym

from parllel.arrays import Array
from parllel.buffers import (Buffer, buffer_asarray, dict_to_namedtuple,
                             namedtuple_to_dict)

from .collections import (EnvInfoType, EnvRandomStepType, EnvSpaces,
                          EnvStepType, ObsType)
from .traj_info import TrajInfo


class Cage(ABC):
    """Cages abstract communication between the sampler and the environments.

    :param EnvClass (Callable): Environment class or factory function
    :param env_kwargs (Dict): Key word arguments that should be passed to the
        `__init__` of `EnvClass` or to the factory function
    :param TrajInfoClass (Callable): TrajectoryInfo class or factory function
    :param reset_automatically (bool): If True (default), environment is reset
    immediately when done is True, replacing the returned observation with the
    reset observation. If False, environment is not reset and the
    `needs_reset` flag is set to True.
    ;param ignore_reset_info (bool): If True (default), the info dictionary 
    returned by env.reset() gets ignored. If False, the dict is treated the
    same way as info dicts returned by env.step() calls
    """

    def __init__(
        self,
        EnvClass: Callable,
        env_kwargs: Dict,
        TrajInfoClass: Callable,
        reset_automatically: bool = True,
        ignore_reset_info: bool = True,
    ) -> None:
        self.EnvClass = EnvClass
        self.env_kwargs = env_kwargs
        self.TrajInfoClass = TrajInfoClass
        self.reset_automatically = reset_automatically
        self.ignore_reset_info = ignore_reset_info

        self._needs_reset: bool = False
        self._render: bool = False

    def _create_env(self) -> None:
        self._completed_trajs: List[TrajInfo] = []
        self._traj_info: TrajInfo = self.TrajInfoClass()

        self._env: gym.Env = self.EnvClass(**self.env_kwargs)
        self._env.reset()

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
    def needs_reset(self) -> bool:
        return self._needs_reset

    @property
    def render(self) -> bool:
        return self._render

    @render.setter
    def render(self, value: bool) -> None:
        self._render = value

    def _step_env(self, action: Buffer) -> EnvStepType:
        # if rendering, render before step is taken so that the renderings
        # line up with the corresponding observation
        if self._render:
            rendering = self._env.render()

        # get underlying numpy arrays and convert to dict if needed
        action = buffer_asarray(action)
        action = namedtuple_to_dict(action)

        obs, reward, terminated, truncated, env_info = self._env.step(action)
        self._traj_info.step(obs, action, reward, terminated, truncated, env_info)

        if self._render:
            env_info["rendering"] = rendering

        return obs, reward, terminated, truncated, env_info

    def _random_step_env(self) -> EnvRandomStepType:
        action = self._env.action_space.sample()
        action = dict_to_namedtuple(action, "action")

        obs, reward, terminated, truncated, env_info = self._step_env(action)

        if terminated or truncated:
            # reset immediately and overwrite last observation
            obs, reset_info = self._reset_env()
            if not self.ignore_reset_info:
                env_info = reset_info

        return action, obs, reward, terminated, truncated, env_info

    def _reset_env(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, EnvInfoType]:
        # store finished trajectory and start new one
        self._completed_trajs.append(self._traj_info)
        self._traj_info = self.TrajInfoClass()
        return self._env.reset(seed=seed, options=options)

    @abstractmethod
    def set_samples_buffer(
        self,
        action: Buffer,
        obs: Buffer,
        reward: Buffer,
        done: Array,
        info: Buffer,
    ) -> None:
        """Declare buffers that will be used during sampling, allowing
        optimized communication during sampling (e.g. with a child process).
        If buffers were passed during cage creation, this is not required.
        """
        raise NotImplementedError

    @abstractmethod
    def step_async(
        self,
        action: Buffer,
        *,
        out_obs: Optional[Buffer] = None,
        out_reward: Optional[Buffer] = None,
        out_terminated: Optional[Buffer] = None,
        out_truncated: Optional[Buffer] = None,
        out_info: Optional[Buffer] = None,
    ) -> None:
        """Step the environment asynchronously using action. If out arguments
        are provided, the result will be written there, otherwise it will be
        returned as a tuple the next time that await_step is called. If any out
        arguments are provided, they must all be. If reset_automatically=True,
        the environment is reset immediately if done=True, and the reset
        observation overwrites the last observation of the trajectory.
        """
        raise NotImplementedError

    @abstractmethod
    def await_step(self) -> EnvStepType | EnvRandomStepType | ObsType:
        """Wait for the asynchronous step to finish and return the results.
        If step_async, reset_async, or random_step_async was called previously
        with input arguments, returns None.
        If step_async was called previously without output arguments, returns
        a tuple of obs, reward, done, and env_info.
        If reset_async was called previously without output arguments, returns
        the reset observation.
        If random_step_async was called previously without output arguments,
        returns a tuple of action, obs, reward, done and env_info.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_completed_trajs(self) -> List[TrajInfo]:
        """Return a list of the TrajInfo objects from trajectories that have
        been completed since the last time this function was called.
        """
        raise NotImplementedError

    @abstractmethod
    def random_step_async(
        self,
        *,
        out_action: Optional[Buffer] = None,
        out_obs: Optional[Buffer] = None,
        out_reward: Optional[Buffer] = None,
        out_terminated: Optional[Buffer] = None,
        out_truncated: Optional[Buffer] = None,
        out_info: Optional[Buffer] = None,
    ) -> None:
        """Take a step with a random action from the env's action space. If
        out arguments are provided, the result will be written there,
        otherwise it will be returned as a tuple the next time that await_step
        is called. If any out arguments are provided, they must all be. The env
        resets automatically if done, regardless of the value of
        reset_automatically.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_async(
        self, *, out_obs: Optional[Buffer] = None, out_info: Optional[Buffer]
    ) -> None:
        """Reset the environment. If out_obs is provided, the reset observation
        is written there, otherwise it is returned the next time that
        await_step is called.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the cage and its environment."""
        raise NotImplementedError
