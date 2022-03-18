from dataclasses import dataclass
import enum
import multiprocessing as mp
from threading import Thread
from typing import Any, List, Optional, Sequence, Tuple, Union

from parllel.arrays import Array
from parllel.buffers import Buffer, NamedArrayTupleClass
from parllel.buffers.registry import BufferRegistry
from parllel.types.traj_info import TrajInfo

from .cage import Cage
from .collections import EnvStep, EnvSpaces


class Command(enum.Enum):
    """Commands for communicating with the subprocess"""
    get_example_output = 0
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


CageSamples = NamedArrayTupleClass("CageSamples", ["action", "observation", "reward", "done", "info"])


class SynchronizedProcessCage(Cage, mp.Process):
    def __init__(self, *args,
        buffers: Optional[Sequence[Buffer]] = None,
        **kwargs,
    ) -> None:
        mp.Process.__init__(self)

        super().__init__(*args, **kwargs)

        # pipe is used for communication between main and child processes
        self._parent_pipe, self._child_pipe = mp.Pipe()

        self._ready = mp.Event()
        self._finished = mp.Event()

        self.buffer_registry = BufferRegistry(buffers)

        # start executing `run` method, which also calls super()._create_env()
        self.start()

        # get env spaces from child process
        self._spaces: EnvSpaces = self._parent_pipe.recv()

        # a simple locking mechanism on the caller side
        # ensures that `step` is always followed by `await_step`
        self._last_command = None

    def _create_env(self, ) -> None:
        # don't create the env yet, we'll do it in the child process
        pass

    def get_example_output(self) -> EnvStep:
        assert self._last_command is None
        self._parent_pipe.send(Message(Command.get_example_output))
        return self._parent_pipe.recv()

    def set_samples_buffer(self, action: Buffer, obs: Buffer, reward: Buffer,
                           done: Array, info: Buffer) -> None:
        """Pass reference to samples buffer after process start."""
        assert self._last_command is None
        samples_buffer = (action, obs, reward, done, info)
        self._parent_pipe.send(Message(Command.register_sample_buffer, samples_buffer))
        for buf in samples_buffer:
            self.buffer_registry.register_buffer(buf)
        self._parent_pipe.recv() # block until finished

    def step_async(self,
        action: Buffer, *,
        out_obs: Buffer = None,
        out_reward: Buffer = None,
        out_done: Buffer = None,
        out_info: Buffer = None
    ) -> None:
        assert self._last_command is None
        # TODO: add assert that these buffers are indexed as expected
        self._finished.clear()
        self._ready.set()
        self._last_command = Command.step

    def _defer_env_reset(self) -> None:
        # execute reset in a separate thread so as not to block batch collection
        assert self._reset_thread is None
        self._reset_thread = Thread(target = super()._defer_env_reset)
        self._reset_thread.start()

    def await_step(self) -> Union[EnvStep, Tuple[Buffer, EnvStep], Buffer]:
        assert self._last_command in {Command.step, Command.random_step, Command.reset_async}
        if self._last_command == Command.step:
            self._last_command = None
            self._finished.wait()
        else:
            self._last_command = None
            return self._parent_pipe.recv()

    def collect_completed_trajs(self) -> List[TrajInfo]:
        assert self._last_command is None
        self._parent_pipe.send(Message(Command.collect_completed_trajs))
        trajs = self._parent_pipe.recv()
        return trajs
    
    def random_step_async(self, *,
        out_action: Buffer = None,
        out_obs: Buffer = None,
        out_reward: Buffer = None,
        out_done: Buffer = None,
        out_info: Buffer = None
    ) -> None:
        """Take a step with a random action from the env's action space.
        """
        assert self._last_command is None
        args = (out_action, out_obs, out_reward, out_done, out_info)
        args = (self.buffer_registry.reduce_buffer(buf) for buf in args)
        self._parent_pipe.send(Message(Command.random_step, tuple(args)))
        self._last_command = Command.random_step
    
    def reset_async(self, out_obs: Buffer = None) -> None:
        assert self._last_command is None
        out_obs = self.buffer_registry.reduce_buffer(out_obs)
        self._parent_pipe.send(Message(Command.reset_async, out_obs))
        self._last_command = Command.reset_async

    def close(self) -> None:
        assert self._last_command is None
        self._parent_pipe.send(Message(Command.close))
        self.join()  # wait for close command to finish
        mp.Process.close(self)

    def run(self):
        """This method runs in a child process. It receives messages through
        follower_pipe, and sends back results.
        """
        super()._create_env() # create env, traj info, etc.

        self._reset_thread: Optional[Thread] = None

        self._closing = False
        self._sampling_thread: Thread = Thread(target=self._sampling_loop)
        self._sampling_thread.start()

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

            if command == Command.get_example_output:
                # data must be None
                env_step: EnvStep = super().get_example_output()
                self._child_pipe.send(env_step)

            elif command == Command.register_sample_buffer:
                samples_buffer = data
                for buf in samples_buffer:
                    self.buffer_registry.register_buffer(buf)
                self.samples_buffer = CageSamples(*samples_buffer)
                self._batch_T = self.samples_buffer.done.shape[0]
                self._t = 0
                self._child_pipe.send(None)

            elif command == Command.collect_completed_trajs:
                # data must be None
                trajs: List[TrajInfo] = super().collect_completed_trajs()
                self._child_pipe.send(trajs)

            elif command == Command.random_step:
                data = (self.buffer_registry.rebuild_buffer(buf) for buf in data)
                out_action, out_obs, out_reward, out_done, out_info = data
                super().random_step_async(out_action=out_action, out_obs=out_obs,
                    out_reward=out_reward, out_done=out_done, out_info=out_info)
                step_result: Tuple[Buffer, EnvStep] = super().await_step()
                self._child_pipe.send(step_result)

            elif command == Command.reset_async:
                out_obs = data
                out_obs = self.buffer_registry.rebuild_buffer(out_obs)
                if self._reset_thread is not None:
                    self._reset_thread.join()
                    self._reset_thread = None
                super().reset_async(out_obs=out_obs)
                reset_obs: Buffer = super().await_step()
                self._child_pipe.send(reset_obs)

            elif command == Command.close:
                self._closing = True
                self._ready.set()  # signal to sampling thread
                self._sampling_thread.join()
                self.buffer_registry.close()
                super().close()  # close Cage object
                break

            else:
                raise ValueError(f"Unhandled command type {command}.")

    def _sampling_loop(self):
        while True:
            self._ready.wait()

            # if shutdown requested, abort sampling and stop thread
            if self._closing:
                break

            # use NamedArrayTuple for convenient slicing
            action, out_obs, out_reward, out_done, out_info = self.samples_buffer[self._t]
            
            # run parent methods synchronously
            super().step_async(action, out_obs=out_obs, out_reward=out_reward,
                               out_done=out_done, out_info=out_info)
            super().await_step()

            # increment time
            self._t = (self._t + 1) % self._batch_T

            # notify caller that result is ready
            self._ready.clear()
            self._finished.set()
