# fmt: off
from __future__ import annotations

import enum
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable

from parllel import Array, ArrayTree

from .cage import Cage
from .collections import (EnvInfoType, EnvRandomStepType, EnvResetType,
                          EnvSpaces, EnvStepType, ObsType)
from .traj_info import TrajInfo


# fmt: on
class Command(enum.Enum):
    """Commands for communicating with the subprocess"""

    step = 2
    collect_completed_trajs = 4
    random_step = 5
    reset_async = 6
    close = 7


@dataclass
class Message:
    """Bundles a command and an optional action for atomic messages"""

    command: Command
    data: Any = None


class ProcessCage(Cage, mp.Process):
    """Environment is created and stepped within a subprocess. Commands are
    sent across Pipes, but data is read from and written directly to the batch
    buffer.

    :param EnvClass (Callable): Environment class or factory function
    :param env_kwargs (Dict): Key word arguments that should be passed to the
        `__init__` of `EnvClass` or to the factory function
    :param TrajInfoClass (Callable): TrajectoryInfo class or factory function
    :param reset_automatically (bool): If True (default), environment is reset
    immediately when done is True, replacing the returned observation with the
    reset observation. If False, environment is not reset and the
    `needs_reset` flag is set to True.
    :param ignore_reset_info (bool): If True (default), the info dictionary
    returned by env.reset() gets ignored. If False, the dict is treated the
    same way as info dicts returned by env.step() calls
    """

    # TODO: add spaces and needs_reset properties

    def __init__(
        self,
        EnvClass: Callable,
        env_kwargs: dict[str, Any],
        TrajInfoClass: Callable,
        reset_automatically: bool = False,
        ignore_reset_info: bool = True,
    ) -> None:
        mp.Process.__init__(self)

        super().__init__(
            EnvClass=EnvClass,
            env_kwargs=env_kwargs,
            TrajInfoClass=TrajInfoClass,
            reset_automatically=reset_automatically,
            ignore_reset_info=ignore_reset_info,
        )

        # pipe is used for communication between main and child processes
        self._parent_pipe, self._child_pipe = mp.Pipe()

        # start executing `run` method, which also creates the environment
        self.start()

        # get env spaces from child process
        self._spaces: EnvSpaces = self._parent_pipe.recv()

        # a simple locking mechanism on the caller side
        # ensures that `step` is always followed by `await_step`
        self.waiting: bool = False

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
        assert not self.waiting
        args = (
            action,
            out_next_obs,
            out_obs,
            out_reward,
            out_terminated,
            out_truncated,
            out_info,
            out_reset_info,
        )
        self._parent_pipe.send(Message(Command.step, args))
        self.waiting = True

    def await_step(self) -> EnvStepType | EnvRandomStepType | EnvResetType | None:
        assert self.waiting
        result = self._parent_pipe.recv()
        self.waiting = False
        if isinstance(result, bool):
            # obs, reward, done, etc. already written to out_args
            self._needs_reset = result
        else:
            self._needs_reset = self._parent_pipe.recv()
            return result

    def collect_completed_trajs(self) -> list[TrajInfo]:
        assert not self.waiting
        self._parent_pipe.send(Message(Command.collect_completed_trajs))
        trajs = self._parent_pipe.recv()
        return trajs

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
        assert not self.waiting
        args = (
            out_action,
            out_next_obs,
            out_obs,
            out_reward,
            out_terminated,
            out_truncated,
            out_info,
            out_reset_info,
        )
        self._parent_pipe.send(Message(Command.random_step, args))
        self.waiting = True

    def reset_async(
        self,
        *,
        out_obs: ArrayTree[Array] | None = None,
        out_info: ArrayTree[Array] | None = None,
    ) -> None:
        assert not self.waiting
        self._parent_pipe.send(Message(Command.reset_async, (out_obs, out_info)))
        self.waiting = True

    def close(self) -> None:
        assert not self.waiting
        self._parent_pipe.send(Message(Command.close))
        self.join()  # wait for close command to finish
        mp.Process.close(self)

    def run(self) -> None:
        """This method runs in a child process. It receives messages through
        child_pipe, and sends back results.
        """
        self._create_env()  # create env, traj info, etc.

        # send env spaces back to parent
        # parent process can receive gym Space objects because gym is imported
        self._child_pipe.send(self._spaces)

        # used to store the result of asynchronous reset
        _reset_obs: ObsType | None = None
        _reset_info: EnvInfoType | None = None

        while True:
            message: Message = self._child_pipe.recv()
            command: Command = message.command
            data: Any = message.data

            if command == Command.step:
                assert not self.needs_reset
                (
                    action,
                    out_next_obs,
                    out_obs,
                    out_reward,
                    out_terminated,
                    out_truncated,
                    out_info,
                    out_reset_info,
                ) = data
                next_obs, reward, terminated, truncated, env_info = self._step_env(
                    action
                )
                obs = next_obs

                if done := terminated or truncated:
                    if self.reset_automatically:
                        # reset immediately and overwrite last observation
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
                    self._child_pipe.send(
                        (
                            next_obs,
                            obs,
                            reward,
                            terminated,
                            truncated,
                            env_info,
                        )
                    )

                self._child_pipe.send(self.needs_reset)

                # this Cage should not be stepped until the end of the batch
                # so we start resetting already
                if done and not self.reset_automatically:
                    _reset_obs, _reset_info = self._reset_env()

            elif command == Command.collect_completed_trajs:
                # data must be None
                trajs = self._completed_trajs
                self._completed_trajs = []
                self._child_pipe.send(trajs)

            elif command == Command.random_step:
                assert not self.needs_reset
                (
                    out_action,
                    out_next_obs,
                    out_obs,
                    out_reward,
                    out_terminated,
                    out_truncated,
                    out_info,
                    out_reset_info,
                ) = data
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
                    self._child_pipe.send(
                        (
                            action,
                            next_obs,
                            obs,
                            reward,
                            terminated,
                            truncated,
                            env_info,
                        )
                    )

                # needs_reset should always be False because we reset if done
                self._child_pipe.send(False)

            elif command == Command.reset_async:
                out_obs, out_info = data
                if _reset_obs is not None and _reset_info is not None:
                    # reset was already carried out in the background
                    reset_obs, reset_info = _reset_obs, _reset_info
                    _reset_obs, _reset_info = None, None
                else:
                    # reset requested when environment was not done
                    reset_obs, reset_info = self._reset_env()

                if out_obs is not None:
                    out_obs[...] = reset_obs
                if out_info is not None:
                    out_info[...] = reset_info
                if out_obs is None and out_info is None:
                    # return step result if user passed no output args at all
                    self._child_pipe.send((reset_obs, reset_info))

                self._needs_reset = False
                self._child_pipe.send(self.needs_reset)

            elif command == Command.close:
                self._close_env()  # close env object
                break

            else:
                raise ValueError(f"Unhandled command type {command}.")
