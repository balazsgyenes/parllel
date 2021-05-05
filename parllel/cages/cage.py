from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from parllel.envs.collections import EnvStep
from parllel.types.traj_info import TrajInfo
from parllel.types.named_tuple import NamedTuple

class Cage:
    """Cages abstract communication between the sampler and the environments.

    Args:
        EnvClass (Callable): TODO
        env_kwargs (Dict): Key word arguments that should be passed to the `__init__` of `EnvClass`
        TrajInfoClass (Callable): TODO
        traj_info_kwargs (Dict): Key word arguments that should be passed to the `__init__` of `TrajInfoClass`
        wait_before_reset (bool): TODO

    TODO: this class will need to own a buffer object of some kind so it can
    write into it. Otherwise there is no way except to pass numpy arrays
    through pipes.
    TODO: is there a way to prevent calls to agent.step() for environments that
    are done and waiting to be reset?
    """
    def __init__(self,
        EnvClass: Callable,
        env_kwargs: Dict,
        TrajInfoClass: Callable,
        traj_info_kwargs: Dict,
        wait_before_reset: bool = False
    ) -> None:
        self.EnvClass = EnvClass
        self.env_kwargs = env_kwargs
        self.TrajInfoCls = TrajInfoCls
        self.traj_info_kwargs = traj_info_kwargs
        self.wait_before_reset = wait_before_reset

    def initialize(self) -> None:
        self._env = self.EnvClass(**self.env_kwargs)
        self._completed_trajs = []
        self._traj_info = self.TrajInfoClass(**self.traj_info_kwargs)
        self._done = False

    def step_async(self, action) -> None:
        if self._done:
            #TODO: do we need any fake values here or can we just ignore them?
            # ensure that done is True for the rest of the batch
            return
        obs, reward, done, env_info = self._env.step(action)
        self._traj_info.step(obs, action, reward, done, env_info)
        if done:
            self._completed_trajs.append(self._traj_info)
            self._traj_info = self.TrajInfoCls(**self.traj_info_kwargs)
            if self.wait_before_reset:
                self._done = True
            else:
                obs = self._env.reset()
        self._step_result = obs, reward, done, env_info

    def await_step(self) -> EnvStep:
        return self._step_result

    def random_step_async(self):
        """Take a step with a random action from the env's action space.
        """
        raise NotImplementedError

    def collect_completed_trajs(self) -> List[TrajInfo]:
        if self._done:
            obs = self._env.reset()

        completed_trajs = self._completed_trajs
        self._completed_trajs = []

        # TODO: how should obs be returned??
        return completed_trajs

    def close(self) -> None:
        self._env.close()
