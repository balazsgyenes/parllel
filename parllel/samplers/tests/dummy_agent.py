from typing import Union

import numpy as np
from nptyping import NDArray

from parllel.buffers import Buffer, NamedTupleClass
from parllel.handlers import Agent, AgentStep


DummyAgentInfo = NamedTupleClass("DummyAgentInfo", ["observation"])


class DummyAgent(Agent):
    def initialize(self, example_inputs, n_states: int) -> AgentStep:
        self._n_states = n_states
        self.reset()
        obs, = example_inputs
        return AgentStep(
            action = 1,
            agent_info = DummyAgentInfo(observation = obs.copy())
        )

    def reset(self) -> None:
        pass
    
    def reset_one(self, env_index: int) -> None:
        pass

    def step(self, observation: Buffer, *, env_ids: Union[int, slice] = slice(None)) -> AgentStep:
        return AgentStep(
            action = observation.copy() * 2,
            agent_info = DummyAgentInfo(observation=observation.copy()),
        )

    def value(self, observation: Buffer, *, env_ids: Union[int, slice] = slice(None)) -> Buffer:
        return observation.copy() * 10


DummyRecurrentAgentInfo = NamedTupleClass("DummyRecurrentAgentInfo", ["observation", "previous_action"])


class DummyRecurrentAgent(Agent):
    def initialize(self, example_inputs, n_states: int) -> AgentStep:
        self._n_states = n_states
        self.reset()
        obs, prev_action, prev_reward = example_inputs
        return AgentStep(
            action = 1,
            agent_info = DummyRecurrentAgentInfo(
                observation = obs.copy(),
                previous_action = prev_action.copy(),
                previous_reward = prev_reward.copy(),
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
            agent_info = DummyRecurrentAgentInfo(
                observation=observation.copy(),
                previous_action=previous_action.copy(),
            ),
        )

    def value(self, observation: Buffer, previous_action: Buffer, *, env_ids: Union[int, slice] = slice(None)) -> Buffer:
        return self._rnn_states[env_ids].copy() * 10