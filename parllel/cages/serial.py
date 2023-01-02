from typing import Callable, Dict, List, Optional, Union

from parllel.arrays import Array
from parllel.buffers import Buffer

from .cage import Cage
from .collections import EnvStepType, EnvRandomStepType, ObsType
from .traj_info import TrajInfo


class SerialCage(Cage):
    """Environment is created and stepped within the local process. This Cage
    can be useful for debugging, or for toy problems.

    :param EnvClass (Callable): Environment class or factory function
    :param env_kwargs (Dict): Key word arguments that should be passed to the
        `__init__` of `EnvClass` or to the factory function
    :param TrajInfoClass (Callable): TrajectoryInfo class or factory function
    :param reset_automatically (bool): If True (default), environment is reset
    immediately when done is True, replacing the returned observation with the
    reset observation. If False, environment is not reset and the
    `needs_reset` flag is set to True.
    """
    def __init__(self,
        EnvClass: Callable,
        env_kwargs: Dict,
        TrajInfoClass: Callable,
        reset_automatically: bool = True,
    ) -> None:
        super().__init__(
            EnvClass=EnvClass,
            env_kwargs=env_kwargs,
            TrajInfoClass=TrajInfoClass,
            reset_automatically=reset_automatically,
        )
        # create env immediately in the local process
        self._create_env()
        self._step_result: Union[EnvStepType, EnvRandomStepType, ObsType, None] = None

    def set_samples_buffer(self,
        action: Buffer,
        obs: Buffer,
        reward: Buffer,
        done: Array,
        info: Buffer,
    ) -> None:
        pass

    def step_async(self,
        action: Buffer, *,
        out_obs: Optional[Buffer] = None,
        out_reward: Optional[Buffer] = None,
        out_done: Optional[Buffer] = None,
        out_info: Optional[Buffer] = None,
    ) -> None:
        obs, reward, done, env_info = self._step_env(action)

        if done:
            if self.reset_automatically:
                # reset immediately and overwrite last observation
                obs = self._reset_env()
            else:
                # store done state
                self._needs_reset = True
    
        if any(out is None for out in (out_obs, out_reward, out_done, out_info)):
            self._step_result = (obs, reward, done, env_info)
        else:
            out_obs[:] = obs
            out_reward[:] = reward
            out_done[:] = done
            out_info[:] = env_info

    def await_step(self) -> Union[EnvStepType, EnvRandomStepType, ObsType]:
        result = self._step_result
        self._step_result = None
        return result

    def collect_completed_trajs(self) -> List[TrajInfo]:
        completed_trajs = self._completed_trajs
        self._completed_trajs = []
        return completed_trajs

    def random_step_async(self, *,
        out_action: Optional[Buffer] = None,
        out_obs: Optional[Buffer] = None,
        out_reward: Optional[Buffer] = None,
        out_done: Optional[Buffer] = None,
        out_info: Optional[Buffer] = None
    ) -> None:
        action, obs, reward, done, env_info = self._random_step_env()

        if any(out is None for out in (out_action, out_obs, out_reward, out_done, out_info)):
            self._step_result = (action, obs, reward, done, env_info)
        else:
            out_action[:] = action
            out_obs[:] = obs
            out_reward[:] = reward
            out_done[:] = done
            out_info[:] = env_info
        
    def reset_async(self, *, out_obs: Optional[Buffer] = None) -> None:
        reset_obs = self._reset_env()
        self._needs_reset = False

        if out_obs is None:
            self._step_result = reset_obs
        else:
            out_obs[:] = reset_obs

    def close(self) -> None:
        self._close_env()
