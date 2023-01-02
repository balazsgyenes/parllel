from dataclasses import dataclass
import enum
import multiprocessing as mp
from typing import Any, List, Optional, Sequence, Union

from parllel.arrays import Array, ManagedMemoryArray
from parllel.buffers import Buffer
from parllel.buffers.registry import BufferRegistry
from parllel.buffers.utils import buffer_all

from .cage import Cage
from .collections import EnvStepType, EnvRandomStepType, ObsType, EnvSpaces
from .traj_info import TrajInfo


class Command(enum.Enum):
    """Commands for communicating with the subprocess"""
    register_sample_buffer = 1
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
    def __init__(self, *args,
        buffers: Optional[Sequence[Buffer]] = None,
        **kwargs,
    ) -> None:
        mp.Process.__init__(self)

        super().__init__(*args, **kwargs)

        # pipe is used for communication between main and child processes
        self._parent_pipe, self._child_pipe = mp.Pipe()

        # buffer registry allows for sending buffers (tuples of arrays) as a
        # buffer ID and an indexing history
        self.buffer_registry = BufferRegistry(buffers)

        # start executing `run` method, which also creates the environment
        self.start()

        # get env spaces from child process
        self._spaces: EnvSpaces = self._parent_pipe.recv()

        # a simple locking mechanism on the caller side
        # ensures that `step` is always followed by `await_step`
        self.waiting: bool = False

    def set_samples_buffer(self,
        action: Buffer,
        obs: Buffer,
        reward: Buffer,
        done: Array,
        info: Buffer,
    ) -> None:
        """Pass reference to samples buffer after process start."""
        assert not self.waiting
        samples_buffer = (action, obs, reward, done, info)
        if not buffer_all(samples_buffer, lambda arr: isinstance(arr, ManagedMemoryArray)):
            raise TypeError(
                "Only ManagedMemoryArray can be set as samples buffer after "
                "process start. Either use ManagedMemoryArrays or pass the "
                "sample buffer to the cage on init."
            )
        self._parent_pipe.send(Message(Command.register_sample_buffer, samples_buffer))
        for buf in samples_buffer:
            self.buffer_registry.register_buffer(buf)
        self._parent_pipe.recv() # block until finished

    def step_async(self,
        action: Buffer, *,
        out_obs: Optional[Buffer] = None,
        out_reward: Optional[Buffer] = None,
        out_done: Optional[Buffer] = None,
        out_info: Optional[Buffer] = None,
    ) -> None:
        assert not self.waiting
        args = (action, out_obs, out_reward, out_done, out_info)
        args = tuple(self.buffer_registry.reduce_buffer(buf) for buf in args)
        self._parent_pipe.send(Message(Command.step, args))
        self.waiting = True

    def await_step(self) -> Union[EnvStepType, EnvRandomStepType, ObsType]:
        assert self.waiting
        result = self._parent_pipe.recv()
        self.waiting = False
        if isinstance(result, bool):
            # obs, reward, done, info already written to out_args
            self._needs_reset = result
        else:
            self._needs_reset = self._parent_pipe.recv()
            return result

    def collect_completed_trajs(self) -> List[TrajInfo]:
        assert not self.waiting
        self._parent_pipe.send(Message(Command.collect_completed_trajs))
        trajs = self._parent_pipe.recv()
        return trajs
    
    def random_step_async(self, *,
        out_action: Optional[Buffer] = None,
        out_obs: Optional[Buffer] = None,
        out_reward: Optional[Buffer] = None,
        out_done: Optional[Buffer] = None,
        out_info: Optional[Buffer] = None
    ) -> None:
        assert not self.waiting
        args = (out_action, out_obs, out_reward, out_done, out_info)
        args = tuple(self.buffer_registry.reduce_buffer(buf) for buf in args)
        self._parent_pipe.send(Message(Command.random_step, args))
        self.waiting = True
    
    def reset_async(self, *, out_obs: Optional[Buffer] = None) -> None:
        assert not self.waiting
        out_obs = self.buffer_registry.reduce_buffer(out_obs)
        self._parent_pipe.send(Message(Command.reset_async, out_obs))
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
        self._create_env() # create env, traj info, etc.

        _reset_obs: Optional[Buffer] = None

        # send env spaces back to parent
        # parent process can receive gym Space objects because gym is imported
        self._child_pipe.send(EnvSpaces(
            observation=self._env.observation_space,
            action=self._env.action_space,
        ))

        while True:
            message: Message = self._child_pipe.recv()
            command: Command = message.command
            data: Any = message.data

            if command == Command.register_sample_buffer:
                samples_buffer = data
                for buf in samples_buffer:
                    self.buffer_registry.register_buffer(buf)
                self._child_pipe.send(None)

            elif command == Command.step:
                data = (self.buffer_registry.rebuild_buffer(buf) for buf in data)
                action, out_obs, out_reward, out_done, out_info = data
                obs, reward, done, env_info = self._step_env(action)

                if done:
                    if self.reset_automatically:
                        # reset immediately and overwrite last observation
                        obs = self._reset_env()
                    else:
                        # store done state
                        self._needs_reset = True

                if any(out is None for out in (out_obs, out_reward, out_done, out_info)):
                    self._child_pipe.send((obs, reward, done, env_info))
                else:
                    out_obs[:] = obs
                    out_reward[:] = reward
                    out_done[:] = done
                    out_info[:] = env_info
                self._child_pipe.send(self.needs_reset)

                # this Cage should not be stepped until the end of the batch
                # so we start resetting already
                if done and not self.reset_automatically:
                    _reset_obs = self._reset_env()

            elif command == Command.collect_completed_trajs:
                # data must be None
                trajs = self._completed_trajs
                self._completed_trajs = []
                self._child_pipe.send(trajs)

            elif command == Command.random_step:
                data = (self.buffer_registry.rebuild_buffer(buf) for buf in data)
                out_action, out_obs, out_reward, out_done, out_info = data
                action, obs, reward, done, env_info = self._random_step_env()

                if any(out is None for out in (out_action, out_obs, out_reward, out_done, out_info)):
                    self._child_pipe.send((action, obs, reward, done, env_info))
                else:
                    out_action[:] = action
                    out_obs[:] = obs
                    out_reward[:] = reward
                    out_done[:] = done
                    out_info[:] = env_info
                # already done is always False because resets automatically
                self._child_pipe.send(False)

            elif command == Command.reset_async:
                out_obs = data
                out_obs = self.buffer_registry.rebuild_buffer(out_obs)
                if _reset_obs is None:
                    reset_obs = _reset_obs
                    _reset_obs = None
                else:
                    reset_obs = self._reset_env()

                if out_obs is None:
                    out_obs[:] = reset_obs
                    self._child_pipe.send(reset_obs)
                # already done is always False after reset
                self._child_pipe.send(False)

            elif command == Command.close:
                self.buffer_registry.close()
                self._close_env()  # close env object
                break

            else:
                raise ValueError(f"Unhandled command type {command}.")
