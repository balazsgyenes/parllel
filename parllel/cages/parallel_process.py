from dataclasses import dataclass
import enum
from functools import partial
import multiprocessing as mp
from typing import Dict, List, Tuple, Callable, Any, Optional, Union

import numpy as np
from nptyping import NDArray

from parllel.buffers import Buffer
from parllel.buffers.reductions import register_buffer_pickler, register_shared_memory_buffer
# from parllel.buffers.weak import WeakBuffer
from parllel.envs.collections import EnvStep, EnvSpaces
# from parllel.types.named_tuple import NamedArrayTuple, NamedTuple
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
        wait_before_reset: bool = False,
    ) -> None:
        mp.Process.__init__(self)

        super().__init__(EnvClass, env_kwargs, TrajInfoCls, traj_info_kwargs,
            wait_before_reset)

    def initialize(self, samples_buffer: Optional[Buffer]) -> None:
        """Instantiate environment and subprocess, etc.
        """
        #TODO: possible to support case where no pre-allocated samples_buffer given?
        self._samples_buffer = samples_buffer

        self._leader_pipe, self._follower_pipe = mp.Pipe()

        # start executing `run` method, which also calls super().initialize()
        self.start()

        register_buffer_pickler()
        register_shared_memory_buffer(self._samples_buffer)

        self._spaces: EnvSpaces = self._leader_pipe.recv()

        # a simple locking mechanism on the caller side
        # ensures that `step` is always followed by `await_step`
        self._last_command = None

    @Cage.spaces.getter
    def spaces(self) -> EnvSpaces:
        return self._spaces

    def step_async(self,
        action: Buffer, *,
        out_obs: Buffer = None,
        out_reward: Buffer = None,
        out_done: Buffer = None,
        out_info: Buffer = None
    ) -> None:
        assert self._last_command is None
        self._leader_pipe.send(Message(Command.step,
            (action, out_obs, out_reward, out_done, out_info)))
        self._last_command = Command.step

    def await_step(self) -> EnvStep:
        assert self._last_command in {Command.step, Command.random_step}
        env_step = self._leader_pipe.recv()
        self._last_command = None
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

        register_buffer_pickler()
        register_shared_memory_buffer(self._samples_buffer)

        self._follower_pipe.send(EnvSpaces(
            observation=self._env.observation_space,
            action=self._env.action_space,
        ))

        while True:
            message = self._follower_pipe.recv()
            command = message.command
            data = message.data

            if command == Command.step:
                action, out_obs, out_reward, out_done, out_info = data
                super().step_async(action, out_obs=out_obs, out_reward=out_reward, out_done=out_done, out_info=out_info)
                step_result: Union[EnvStep, object] = super().await_step()
                self._follower_pipe.send(step_result)

            elif command == Command.random_step:
                out_obs, out_reward, out_done, out_info = data
                super().random_step_async(out_obs=out_obs, out_reward=out_reward, out_done=out_done, out_info=out_info)
                step_result: Union[EnvStep, object] = super().await_step()
                self._follower_pipe.send(step_result)

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
