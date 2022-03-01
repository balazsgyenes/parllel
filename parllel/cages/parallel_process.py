from dataclasses import dataclass
import enum
import multiprocessing as mp
from threading import Thread
from typing import List, Tuple, Any, Optional, Union

from parllel.buffers import Buffer
from parllel.buffers.reductions import register_buffer_pickler, register_shared_memory_buffer
from parllel.envs.collections import EnvStep, EnvSpaces
from parllel.types.traj_info import TrajInfo
from .cage import Cage


class Command(enum.Enum):
    """Commands for communicating with the subprocess"""
    get_example_output = 0
    set_sample_buffer = 1
    step = 2
    collect_deferred_reset = 3
    collect_completed_trajs = 4
    random_step = 5
    reset_async = 6
    close = 7


@dataclass
class Message:
    """Bundles a command and an optional action for atomic messages"""
    command: Command
    data: Any = None


class ParallelProcessCage(Cage, mp.Process):
    def __init__(self, *args,
        samples_buffer: Optional[Buffer] = None,
        **kwargs,
    ) -> None:
        mp.Process.__init__(self)

        super().__init__(*args, **kwargs)
        self._samples_buffer = samples_buffer

        # pipe is used for communication between main and child processes
        self._parent_pipe, self._child_pipe = mp.Pipe()

        # start executing `run` method, which also calls super().initialize()
        self.start()

        # register custom methods for sending buffer objects between processes
        register_buffer_pickler()
        if self._samples_buffer is not None:
            # enable receiving buffers derived from samples buffer
            register_shared_memory_buffer(self._samples_buffer)

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

    def set_samples_buffer(self, samples_buffer: Buffer) -> None:
        """Pass reference to samples buffer after process start."""
        assert self._last_command is None
        self._samples_buffer = samples_buffer

        self._parent_pipe.send(Message(Command.set_sample_buffer, samples_buffer))
        register_shared_memory_buffer(self._samples_buffer) # register on parent side
        self._parent_pipe.recv() # block until finished

    def step_async(self,
        action: Buffer, *,
        out_obs: Buffer = None,
        out_reward: Buffer = None,
        out_done: Buffer = None,
        out_info: Buffer = None
    ) -> None:
        assert self._last_command is None
        self._parent_pipe.send(Message(Command.step,
            (action, out_obs, out_reward, out_done, out_info)))
        self._last_command = Command.step

    def _defer_env_reset(self) -> None:
        # execute reset in a separate thread so as not to block batch collection
        self._reset_thread = Thread(target = super()._defer_env_reset)
        self._reset_thread.start()

    def await_step(self) -> Union[EnvStep, Tuple[Buffer, EnvStep], Buffer]:
        assert self._last_command in {Command.step, Command.random_step, Command.reset_async}
        self._last_command = None
        return self._parent_pipe.recv()

    def collect_deferred_reset(self, *, out_obs: Buffer = None) -> Optional[Buffer]:
        assert self._last_command is None and self.wait_before_reset
        self._parent_pipe.send(Message(Command.collect_deferred_reset, out_obs))
        self._last_command = Command.collect_deferred_reset
    
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
        self._parent_pipe.send(Message(Command.random_step,
            (out_action, out_obs, out_reward, out_done, out_info)))
        self._last_command = Command.random_step
    
    def reset_async(self, out_obs: Buffer = None) -> None:
        assert self._last_command is None
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

        # do the identical setup in child process as in parent
        register_buffer_pickler()
        if self._samples_buffer is not None:
            register_shared_memory_buffer(self._samples_buffer)

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

            elif command == Command.set_sample_buffer:
                self._samples_buffer = data
                register_shared_memory_buffer(self._samples_buffer)
                self._child_pipe.send(None)

            elif command == Command.step:
                action, out_obs, out_reward, out_done, out_info = data
                super().step_async(action, out_obs=out_obs, out_reward=out_reward,
                    out_done=out_done, out_info=out_info)
                step_result: EnvStep = super().await_step()
                self._child_pipe.send(step_result)

            elif command == Command.collect_deferred_reset:
                out_obs = data
                self._reset_thread.join()
                reset_obs: Buffer = super().collect_deferred_reset(out_obs=out_obs)
                self._child_pipe.send(reset_obs)

            elif command == Command.collect_completed_trajs:
                # data must be None
                trajs: List[TrajInfo] = super().collect_completed_trajs()
                self._child_pipe.send(trajs)

            elif command == Command.random_step:
                out_action, out_obs, out_reward, out_done, out_info = data
                super().random_step_async(out_action=out_action, out_obs=out_obs,
                    out_reward=out_reward, out_done=out_done, out_info=out_info)
                step_result: Tuple[Buffer, EnvStep] = super().await_step()
                self._child_pipe.send(step_result)

            elif command == Command.reset_async:
                out_obs = data
                super().reset_async(out_obs=out_obs)
                reset_obs: Buffer = super().await_step()
                self._child_pipe.send(reset_obs)

            elif command == Command.close:
                super().close()  # close Cage object
                break

            else:
                raise ValueError(f"Unhandled command type {command}.")
