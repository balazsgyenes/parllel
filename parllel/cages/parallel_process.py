from dataclasses import dataclass
import enum
from typing import Dict, List, Callable, Any
import multiprocessing as mp

from parllel.envs.collections import EnvStep
from parllel.types.traj_info import TrajInfo
from .cage import Cage


class Command(enum.Enum):
    """Commands for communicating with the subprocess"""
    step = 1
    random_step = 2
    collect_completed_trajs = 3
    close = 4


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
        
    def initialize(self) -> None:
        """Instantiate environment and subprocess, etc.
        """
        self._leader_pipe, self._follower_pipe = mp.Pipe()
        # start executing `run` method, which also calls super().initialize()
        self.start()

        # a primitive locking mechanism on the caller side
        # ensures that `step` is always followed by `await_step`
        self._last_command = None

    def step(self, action) -> None:
        assert self._last_command is None
        self._leader_pipe.send(Message(Command.step, action))
        #TODO: obviously we don't want to send this numpy array through a pipe
        self._last_command = Command.step

    def await_step(self) -> EnvStep:
        assert self._last_command in {Command.step, Command.random_step}
        env_step = self._leader_pipe.recv()
        self._last_command = None
        return env_step

    def random_step(self):
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

        self._closing = False
        while not self._closing:
            message = self._follower_pipe.recv()
            command = message.command
            data = message.data

            if command == Command.step:
                # data must be `action` argument
                super().step(data)
                env_step = super().await_step()
                # return must be `EnvStep`
                self._follower_pipe.send(env_step)

            elif command == Command.random_step:
                # data must be None
                super().random_step()
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
                self._closing = True  # TODO: replace with break?

            else:
                raise ValueError(f"Unhandled command type {command}.")
