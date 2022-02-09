from typing import Union

import numpy as np
from nptyping import NDArray

from parllel.buffers import Buffer, NamedTupleClass, buffer_method
from parllel.handlers import Agent, AgentStep


DummyAgentInfo = NamedTupleClass("DummyAgentInfo", ["observation", "previous_action"])


class DummyAgent(Agent):
    def initialize(self, example_inputs, n_states: int) -> AgentStep:
        self._n_states = n_states
        self.reset()
        observation, previous_action = example_inputs
        return AgentStep(
            action = 1,
            agent_info = DummyAgentInfo(
                observation = buffer_method(observation, "copy"),
                previous_action = buffer_method(previous_action, "copy"),
            )
        )

    def reset(self) -> None:
        self._rnn_states: NDArray = np.zeros(shape=(self._n_states,), dtype=np.int32)
    
    def reset_one(self, env_index: int) -> None:
        self._rnn_states[env_index] = 0

    def step(self, observation: Buffer, previous_action: Buffer, *, env_ids: Union[int, slice] = slice(None)) -> AgentStep:
        self._rnn_states[env_ids] += 1
        return AgentStep(
            action = self._rnn_states[env_ids].copy(),
            agent_info = DummyAgentInfo(
                observation = buffer_method(observation, "copy"),
                previous_action = buffer_method(previous_action, "copy"),
            ),
        )

    def value(self, observation: Buffer, previous_action: Buffer, *, env_ids: Union[int, slice] = slice(None)) -> Buffer:
        return self._rnn_states[env_ids].copy() * 10