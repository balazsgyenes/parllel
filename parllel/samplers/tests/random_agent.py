from typing import Union

import gym
import numpy as np
from nptyping import NDArray

from parllel.buffers import Buffer, NamedTupleClass, buffer_method
from parllel.handlers import Agent, AgentStep


RandomAgentInfo = NamedTupleClass("RandomAgentInfo", ["observation"])
RandomRecurrentAgentInfo = NamedTupleClass("RandomAgentInfo", ["observation", "previous_action"])


class RandomAgent(Agent):
    def __init__(self, action_space: gym.Space, recurrent: bool = False) -> None:
        super().__init__()
        self.action_space = action_space
        self.recurrent = recurrent
        self.InfoCls = RandomRecurrentAgentInfo if recurrent else RandomAgentInfo
    
    def initialize(self, example_inputs, n_states: int) -> AgentStep:
        self._n_states = n_states
        self.reset()
        observation, previous_action = example_inputs
        agent_info = (buffer_method(observation, "copy"),)
        if self.recurrent:
            agent_info += (buffer_method(previous_action, "copy"),)
        return AgentStep(
            action = self.action_space.sample(),
            agent_info = self.InfoCls(*agent_info)
        )

    def reset(self) -> None:
        self._rnn_states: NDArray = np.zeros(shape=(self._n_states,), dtype=np.int32)
    
    def reset_one(self, env_index: int) -> None:
        self._rnn_states[env_index] = 0

    def step(self, observation: Buffer, previous_action: Buffer, *, env_ids: Union[int, slice] = slice(None)) -> AgentStep:
        action = np.asarray([self.action_space.sample() for _ in self._rnn_states[env_ids]])
        self._rnn_states[env_ids] += 1
        agent_info = (buffer_method(observation, "copy"),)
        if self.recurrent:
            agent_info += (buffer_method(previous_action, "copy"),)
        return AgentStep(
            action = action,
            agent_info = self.InfoCls(*agent_info),
        )

    def value(self, observation: Buffer, previous_action: Buffer, *, env_ids: Union[int, slice] = slice(None)) -> Buffer:
        return self._rnn_states[env_ids].copy() * 10