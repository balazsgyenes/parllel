from __future__ import annotations

from typing import Any, Callable, Mapping

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
    """

    def __init__(
        self,
        EnvClass: Callable,
        env_kwargs: Mapping[str, Any] | None = None,
        TrajInfoClass: Callable | None = None,
        reset_automatically: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            EnvClass=EnvClass,
            env_kwargs=env_kwargs,
            TrajInfoClass=TrajInfoClass,
            reset_automatically=reset_automatically,
            seed=seed,
        )
        # create env immediately in the local process
        self._create_env()
        self._step_result: EnvStepType | EnvRandomStepType | EnvResetType | None = None

    def step_async(
        self,
        action: ArrayTree[Array],
        *,
        out_next_obs: ArrayTree[Array] | None = None,
        out_obs: ArrayTree[Array] | None = None,
        out_reward: ArrayTree[Array] | None = None,
        out_terminated: Array | None = None,
        out_truncated: Array | None = None,
        out_info: ArrayTree[Array] | None = None,
        out_reset_info: ArrayTree[Array] | None = None,
    ) -> None:
        next_obs, reward, terminated, truncated, env_info = self._step_env(action)
        obs = next_obs

        if terminated or truncated:
            if self.reset_automatically:
                # reset immediately and overwrite last observation and info
                obs, reset_info = self._reset_env()
                if out_reset_info is not None:
                    out_reset_info[...] = reset_info
            else:
                # store done state
                self._needs_reset = True

        out_pairs = (
            (out_next_obs, next_obs),
            (out_obs, obs),
            (out_reward, reward),
            (out_terminated, terminated),
            (out_truncated, truncated),
            (out_info, env_info),
        )

        if pairs_to_write := [
            (out, result) for out, result in out_pairs if out is not None
        ]:
            for out, result in pairs_to_write:
                out[...] = result
        else:
            # return step result if user passed no output args at all
            self._step_result = (
                next_obs,
                obs,
                reward,
                terminated,
                truncated,
                env_info,
            )

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
        out_next_obs: ArrayTree[Array] | None = None,
        out_obs: ArrayTree[Array] | None = None,
        out_reward: ArrayTree[Array] | None = None,
        out_terminated: Array | None = None,
        out_truncated: Array | None = None,
        out_info: ArrayTree[Array] | None = None,
        out_reset_info: ArrayTree[Array] | None = None,
    ) -> None:
        (
            action,
            next_obs,
            reward,
            terminated,
            truncated,
            env_info,
        ) = self._random_step_env()
        obs = next_obs

        if terminated or truncated:
            # reset immediately and overwrite last observation
            obs, reset_info = self._reset_env()
            if out_reset_info is not None:
                out_reset_info[...] = reset_info

        out_pairs = (
            (out_action, action),
            (out_next_obs, next_obs),
            (out_obs, obs),
            (out_reward, reward),
            (out_terminated, terminated),
            (out_truncated, truncated),
            (out_info, env_info),
        )

        if pairs_to_write := [
            (out, result) for out, result in out_pairs if out is not None
        ]:
            for out, result in pairs_to_write:
                out[...] = result
        else:
            # return step result if user passed no output args at all
            self._step_result = (
                action,
                next_obs,
                obs,
                reward,
                terminated,
                truncated,
                env_info,
            )

    def reset_async(
        self,
        *,
        out_obs: ArrayTree[Array] | None = None,
        out_info: ArrayTree[Array] | None = None,
    ) -> None:
        reset_obs, reset_info = self._reset_env()
        self._needs_reset = False

        if out_obs is not None:
            out_obs[...] = reset_obs
        if out_info is not None:
            out_info[...] = reset_info
        if out_obs is None and out_info is None:
            # return step result if user passed no output args at all
            self._step_result = (reset_obs, reset_info)

    def get_attr(self, name: str) -> Any:
        return getattr(self._env, name)

    def set_attr(self, name: str, value: Any) -> None:
        return setattr(self._env, name, value)

    def close(self) -> None:
        self._close_env()
