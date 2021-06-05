from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from nptyping import NDArray
import gym

from parllel.envs.collections import EnvStep, EnvSpaces
from parllel.types.traj_info import TrajInfo
from parllel.buffers import Buffer

class Cage:
    """Cages abstract communication between the sampler and the environments.

    Note: a literal empty tuple `()` is used in place of None where an error
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

    TODO: prevent calls to agent.step() for environments that are done and waiting to be reset
    TODO: can we assume that the cage will not be called if it is already done? Can the code be more streamlined for this case?
    """
    def __init__(self,
        EnvClass: Callable,
        env_kwargs: Dict,
        TrajInfoClass: Callable,
        traj_info_kwargs: Dict,
        wait_before_reset: bool = False,
    ) -> None:
        self.EnvClass = EnvClass
        self.env_kwargs = env_kwargs
        self.TrajInfoClass = TrajInfoClass
        self.traj_info_kwargs = traj_info_kwargs
        self.wait_before_reset = wait_before_reset

    def initialize(self) -> None:
        self._env: gym.Env = self.EnvClass(**self.env_kwargs)
        self._env.reset()
        self._completed_trajs: List[TrajInfo] = []
        self._traj_info: TrajInfo = self.TrajInfoClass(**self.traj_info_kwargs)
        self._already_done: bool = False
        self._step_result: Union[EnvStep, tuple] = ()
        self._reset_obs: Union[NDArray, Tuple] = ()

    @property
    def spaces(self) -> EnvSpaces:
        return EnvSpaces(
            observation=self._env.observation_space,
            action=self._env.action_space,
        )

    def step_async(self,
        action: Buffer, *,
        out_obs: Buffer = None,
        out_reward: Buffer = None,
        out_done: Buffer = None,
        out_info: Buffer = None,
    ) -> None:
        """If any out parameter is given, they must all be given. 
        """
        if self._already_done:
            if any(out is None for out in (out_obs, out_reward, out_done, out_info)):
                # Nones are ignored by NamedArrayTuple
                # done must be set to True in all cases though
                self._step_result = EnvStep(None, None, True, None) 
            else:
                # leave other values unchanged, since they are ignored anyway
                out_done[:] = True
                # TODO: write zeros to other buffers, because zeros are faster to compute with?
                # out_obs[:] = 0
                # out_reward[:] = 0
                # out_info[:] = 0
            return

        obs, reward, done, env_info = self._env.step(np.asarray(action))
        self._traj_info.step(obs, action, reward, done, env_info)
        
        if done:
            # store finished trajectory and start new one
            self._completed_trajs.append(self._traj_info)
            self._traj_info = self.TrajInfoClass(**self.traj_info_kwargs)
            if self.wait_before_reset:
                # start environment reset, maybe asynchronously
                self._reset_env_async()
                # store done state
                self._already_done = True
            else:
                # reset immediately and overwrite last observation
                obs = self._env.reset()
    
        if any(out is None for out in (out_obs, out_reward, out_done, out_info)):
            self._step_result = EnvStep(obs, reward, done, env_info)
        else:
            out_obs[:] = obs  # TODO: no assignment possible if out_obs is 0D-array
            out_reward[:] = reward
            out_done[:] = done
            out_info[:] = env_info

    def await_step(self) -> Optional[EnvStep]:
        result = self._step_result
        self._step_result = ()
        return result

    def random_step_async(self, *,
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

    @property
    def already_done(self) -> bool:
        return self._already_done

    def collect_completed_trajs(self) -> List[TrajInfo]:
        self._already_done = False

        completed_trajs = self._completed_trajs
        self._completed_trajs = []

        return completed_trajs

    def _reset_env_async(self) -> None:
        self._reset_obs = self._env.reset()

    def collect_reset_obs(self, *, out_obs: Buffer = None) -> Optional[NDArray]:
        result = self._reset_obs
        self._reset_obs = ()
        if out_obs is not None:
            out_obs[:] = result
        else:
            return result

    def close(self) -> None:
        self._env.close()
