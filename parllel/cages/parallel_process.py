from dataclasses import dataclass
from functools import partial
import enum
from typing import Dict, List, Callable, Any
import multiprocessing as mp

from parllel.buffers.weak import WeakBuffer
from parllel.envs.collections import EnvStep
from parllel.types.traj_info import TrajInfo
from .cage import Cage


class Command(enum.Enum):
    """Commands for communicating with the subprocess"""
    step = 1
    random_step = 2
    collect_completed_trajs = 3
    close = 4
    write_to_buffer = 5


@dataclass
class Message:
    """Bundles a command and an optional action for atomic messages"""
    command: Command
    data: Any = None


class ParallelProcessCage(Cage, mp.Process):
    def __init__(self,
        EnvClass: Callable,
        env_kwargs: Dict,
        TrajInfoCls: Callable,
        traj_info_kwargs: Dict,
        wait_before_reset: bool = False
    ) -> None:
        mp.Process.__init__(self)

        super().__init__(EnvClass, env_kwargs, TrajInfoCls, traj_info_kwargs,
            wait_before_reset)
        
    def initialize(self, samples_buffer) -> None:
        """Instantiate environment and subprocess, etc.
        """
        self._samples_buffer = samples_buffer
        self._buffer_index = {}

        self._leader_pipe, self._follower_pipe = mp.Pipe()
        # start executing `run` method, which also calls super().initialize()
        self.start()

        def write_callback(target_buffer_id, location, source_array_id):
            self._leader_pipe.send(Message(Command.write_to_buffer, dict(
                target_buffer_id = target_buffer_id,
                location = location,
                source_array_id = source_array_id,
            )))
        self._write_callback = write_callback

        # a primitive locking mechanism on the caller side
        # ensures that `step` is always followed by `await_step`
        self._last_command = None

    def step_async(self, action) -> None:
        assert self._last_command is None
        self._leader_pipe.send(Message(Command.step, action))
        #TODO: obviously we don't want to send this numpy array through a pipe
        self._last_command = Command.step

    def await_step(self) -> EnvStep:
        assert self._last_command in {Command.step, Command.random_step}
        array_ids = self._leader_pipe.recv()
        self._last_command = None

        env_step = EnvStep(WeakBuffer(
                write_callback=partial(self._write_callback, source_buffer_id=array_id),
                read_callback=None,
            ) for array_id in array_ids
        )

        return env_step

    def random_step_async(self):
        """Take a step with a random action from the env's action space.
        """
        assert self._last_command is None
        self._leader_pipe.send(Message(Command.random_step))
        self._last_command = Command.random_step
    
    def collect_completed_trajs(self) -> List[TrajInfo]:
        assert self._last_command is None
        self._leader_pipe.send(Message(Command.collect_completed_trajs))
        trajs = self._leader_pipe.recv()
        return trajs

    def close(self) -> None:
        assert self._last_command is None
        self._leader_pipe.send(Message(Command.close))
        self.join()  # wait for close command to finish
        mp.Process.close(self)

    def run(self):
        """This method runs in a child process. It receives messages through
        follower_pipe, and sends back results.
        """
        # initialize Cage object
        super().initialize()

        _closing = False
        while not _closing:
            message = self._follower_pipe.recv()
            command = message.command
            data = message.data

            if command == Command.step:
                # data must be `action` argument
                super().step_async(data)
                env_step = super().await_step()
                # return must be `EnvStep`

                array_cache = {
                    id(elem): elem
                    for elem in env_step
                }

                self._follower_pipe.send(tuple(array_cache.keys()))

                for _ in range(len(array_cache)):
                    message = self._follower_pipe.recv()
                    assert message.command == Command.write_to_buffer
                    write_call = message.data

                    source_array = array_cache[write_call["source_buffer_id"]]
                    target_buffer = self._buffer_index[write_call["target_buffer_id"]]
                    location = write_call["location"]
                    target_buffer[location] = source_array
            
            elif command == Command.random_step:
                # data must be None
                super().random_step_async()
                env_step = super().await_step()
                # return must be `EnvStep`
                self._follower_pipe.send(env_step)

            elif command == Command.collect_completed_trajs:
                # data must be None
                trajs = super().collect_completed_trajs()
                # trajs must be List[TrajInfo]
                self._follower_pipe.send(trajs)

            elif command == Command.close:
                super().close()  # close Cage object
                _closing = True  # TODO: replace with break?

            else:
                raise ValueError(f"Unhandled command type {command}.")
