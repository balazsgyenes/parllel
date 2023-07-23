from __future__ import annotations

from typing import Any, Callable

from parllel import Array, ArrayTree

from .cage import Cage
from .collections import EnvRandomStepType, EnvResetType, EnvStepType
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
    ;param ignore_reset_info (bool): If True (default), the info dictionary
    returned by env.reset() gets ignored. If False, the dict is treated the
    same way as info dicts returned by env.step() calls
    """

    def __init__(
        self,
        EnvClass: Callable,
        env_kwargs: dict[str, Any],
        TrajInfoClass: Callable,
        reset_automatically: bool = True,
        ignore_reset_info: bool = True,
    ) -> None:
        super().__init__(
            EnvClass=EnvClass,
            env_kwargs=env_kwargs,
            TrajInfoClass=TrajInfoClass,
            reset_automatically=reset_automatically,
            ignore_reset_info=ignore_reset_info,
        )
        # create env immediately in the local process
        self._create_env()
        self._step_result: EnvStepType | EnvRandomStepType | EnvResetType | None = None

    def step_async(
        self,
        action: ArrayTree[Array],
        *,
        out_obs: ArrayTree[Array] | None = None,
        out_reward: ArrayTree[Array] | None = None,
        out_terminated: Array | None = None,
        out_truncated: Array | None = None,
        out_info: ArrayTree[Array] | None = None,
    ) -> None:
        obs, reward, terminated, truncated, env_info = self._step_env(action)

        if terminated or truncated:
            if self.reset_automatically:
                # reset immediately and overwrite last observation and info
                obs, reset_info = self._reset_env()
                if not self.ignore_reset_info:
                    env_info = reset_info
            else:
                # store done state
                self._needs_reset = True

        try:
            out_obs[...] = obs
            out_reward[...] = reward
            out_terminated[...] = terminated
            out_truncated[...] = truncated
            out_info[...] = env_info
        except TypeError as e:
            outs = (
                out_obs,
                out_reward,
                out_terminated,
                out_truncated,
                out_info,
            )
            if any(out is None for out in outs):
                if not all(out is None for out in outs):
                    # if user passed a combination of None and Array, it's probably a mistake
                    raise ValueError(
                        f"Missing {outs.index(None) + 1}nth output argument!"
                    )

                # return step result if user passed no output args at all
                self._step_result = (obs, reward, terminated, truncated, env_info)
            else:
                # otherwise this was an unexpected error
                raise e

    def await_step(self) -> EnvStepType | EnvRandomStepType | EnvResetType | None:
        result = self._step_result
        self._step_result = None
        return result

    def collect_completed_trajs(self) -> list[TrajInfo]:
        completed_trajs = self._completed_trajs
        self._completed_trajs = []
        return completed_trajs

    def random_step_async(
        self,
        *,
        out_action: ArrayTree[Array] | None = None,
        out_obs: ArrayTree[Array] | None = None,
        out_reward: ArrayTree[Array] | None = None,
        out_terminated: Array | None = None,
        out_truncated: Array | None = None,
        out_info: ArrayTree[Array] | None = None,
    ) -> None:
        (
            action,
            obs,
            reward,
            terminated,
            truncated,
            env_info,
        ) = self._random_step_env()

        try:
            out_action[...] = action
            out_obs[...] = obs
            out_reward[...] = reward
            out_terminated[...] = terminated
            out_truncated[...] = truncated
            out_info[...] = env_info
        except TypeError as e:
            outs = (
                out_action,
                out_obs,
                out_reward,
                out_terminated,
                out_truncated,
                out_info,
            )
            if any(out is None for out in outs):
                if not all(out is None for out in outs):
                    # if user passed a combination of None and Array, it's probably a mistake
                    raise ValueError(
                        f"Missing {outs.index(None) + 1}nth output argument!"
                    )

                # return step result if user passed no output args at all
                self._step_result = (
                    action,
                    obs,
                    reward,
                    terminated,
                    truncated,
                    env_info,
                )
            else:
                # otherwise this was an unexpected error
                raise e

    def reset_async(
        self,
        *,
        out_obs: ArrayTree[Array] | None = None,
        out_info: ArrayTree[Array] | None = None,
    ) -> None:
        reset_obs, reset_info = self._reset_env()
        self._needs_reset = False

        try:
            out_obs[...] = reset_obs
            if not self.ignore_reset_info:
                out_info[...] = reset_info
        except TypeError as e:
            outs = (out_obs, out_info)
            if any(out is None for out in outs):
                if not all(out is None for out in outs):
                    # if user passed a combination of None and Array, it's probably a mistake
                    raise ValueError(
                        f"Missing {outs.index(None) + 1}nth output argument!"
                    )

                # return step result if user passed no output args at all
                self._step_result = (reset_obs, reset_info)
            else:
                # otherwise this was an unexpected error
                raise e

    def close(self) -> None:
        self._close_env()
