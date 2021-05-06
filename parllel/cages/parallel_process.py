from dataclasses import dataclass
import enum
from functools import partial
import multiprocessing as mp
from typing import Dict, List, Tuple, Callable, Any, Optional, Union

import numpy as np
from nptyping import NDArray

from parllel.buffers.sharedmemory import SharedMemoryBuffer
from parllel.buffers.weak import WeakBuffer
from parllel.envs.collections import EnvStep
from parllel.types.named_tuple import NamedArrayTuple, NamedTuple
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


@dataclass
class ArrayReference:
    array_id: int


class ParallelProcessCage(Cage, mp.Process):
    def __init__(self,
        EnvClass: Callable,
        env_kwargs: Dict,
        TrajInfoCls: Callable,
        traj_info_kwargs: Dict,
        wait_before_reset: bool = False,
        arrays_not_to_pipe: Optional[List[str]] = None,
    ) -> None:
        mp.Process.__init__(self)

        super().__init__(EnvClass, env_kwargs, TrajInfoCls, traj_info_kwargs,
            wait_before_reset)

        self._names_to_withhold = arrays_not_to_pipe
        
    def initialize(self, samples_buffer: Optional[NamedArrayTuple]) -> None:
        """Instantiate environment and subprocess, etc.
        """
        #TODO: ensure support for when pre-allocated samples buffer given
        self._samples_buffer = samples_buffer
        self._buffer_index = create_buffer_index(self._samples_buffer)

        self._leader_pipe, self._follower_pipe = mp.Pipe()
        #TODO: create queue (or separate pipe) for dispatched writes

        # start executing `run` method, which also calls super().initialize()
        self.start()

        def write_callback(target_buffer_id, location, source_array_id):
            self._leader_pipe.send(Message(Command.write_to_buffer, dict(
                target_buffer_id = target_buffer_id,
                location = location,
                source_array_id = source_array_id,
            )))
        self._write_callback = write_callback

        # a simple locking mechanism on the caller side
        # ensures that `step` is always followed by `await_step`
        self._last_command = None

    def step_async(self, action) -> None:
        assert self._last_command is None
        self._leader_pipe.send(Message(Command.step, action))
        #TODO: provide option to avoid sending this array over the pipe
        # is it possible to retrieve the array reference and the indices?
        self._last_command = Command.step

    def await_step(self) -> EnvStep:
        assert self._last_command in {Command.step, Command.random_step}
        env_step = self._leader_pipe.recv()
        self._last_command = None

        env_step = add_weak_buffers_to_namedtup(
            namedtup=env_step,
            write_callback=self._write_callback,
            read_callback=None,  # TODO: add read callback for debugging
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

        while True:
            message = self._follower_pipe.recv()
            command = message.command
            data = message.data

            if command == Command.step:
                # data must be `action` argument
                super().step_async(data)
                env_step = super().await_step()
                # return must be `EnvStep`
                self.transfer_env_step(env_step)
            
            elif command == Command.random_step:
                # data must be None
                super().random_step_async()
                env_step = super().await_step()
                # return must be `EnvStep`
                self.transfer_env_step(env_step)

            elif command == Command.collect_completed_trajs:
                # data must be None
                trajs = super().collect_completed_trajs()
                # trajs must be List[TrajInfo]
                self._follower_pipe.send(trajs)

            elif command == Command.close:
                super().close()  # close Cage object
                break

            else:
                raise ValueError(f"Unhandled command type {command}.")

    def transfer_env_step(self, env_step):
        env_step, array_cache = withhold_arrays(
            namedtup=env_step,
            array_cache={},
            names_to_withhold=self._names_to_withhold,
        )

        self._follower_pipe.send(env_step)

        for _ in range(len(array_cache)):
            message = self._follower_pipe.recv()
            assert message.command == Command.write_to_buffer
            write_call = message.data

            source_array = array_cache[write_call["source_buffer_id"]]
            target_buffer = self._buffer_index[write_call["target_buffer_id"]]
            location = write_call["location"]
            target_buffer[location] = source_array


def create_buffer_index(
    samples_buffer: Union[None, NamedArrayTuple, SharedMemoryBuffer],
    index: Dict,
) -> Dict[int, SharedMemoryBuffer]:
    if samples_buffer is None:
        pass
    elif isinstance(samples_buffer, SharedMemoryBuffer):
        index[samples_buffer.unique_id] = samples_buffer
    else:
        for elem in samples_buffer:
            create_buffer_index(elem, index)
    return index


def withhold_arrays(
    sample: Union[NamedTuple, NDArray, float, bool],
    array_cache: Dict,
    names_to_withhold: Optional[List[str]],
    name: str = "",
) -> Tuple[NamedTuple, Dict[int, NDArray]]:
    if not names_to_withhold:
        return sample, array_cache
    else:
        raise NotImplementedError

def add_weak_buffers_to_namedtup(
    namedtup: Union[NamedTuple, NDArray, ArrayReference],
    write_callback: Callable,
    read_callback: Callable,
) -> Union[NamedTuple, WeakBuffer]:
    if isinstance(namedtup, np.ndarray):
        # numpy arrays are iterable but we do not want to change them
        return namedtup
    try:
        iterator = iter(namedtup)
    except TypeError:
        # not iterable
        if isinstance(namedtup, ArrayReference):
            # convert array references to weak buffers
            return WeakBuffer(
                write_callback=partial(write_callback, source_buffer_id=namedtup.array_id),
                read_callback=read_callback,
            )
        else:
            # otherwise just return the object
            return namedtup
    else:
        # iterable
        elems = (add_weak_buffers_to_namedtup(elem, write_callback, read_callback)
            for elem in namedtup)
        return namedtup._make(elems)
