from dataclasses import dataclass
from typing import Union

import numpy as np
from nptyping import NDArray

from parllel.buffers import Buffer
from parllel.handlers import Agent, AgentStep


@dataclass
class DummyAgentInfo:
    observation: Buffer
    previous_action: Buffer
    previous_reward: Buffer


class DummyAgent(Agent):
    def initialize(self, n_states: int) -> None:
        self._n_states = n_states
        self.reset()

    def reset(self) -> None:
        self._rnn_states: NDArray = np.zeros(shape=(self._n_states,), dtype=np.int32)
    
    def reset_one(self, env_index: int) -> None:
        self._rnn_states[env_index] = 0

    def step(self, observation: Buffer, previous_action: Buffer, previous_reward: Buffer, *, env_ids: Union[int, slice]) -> AgentStep:
        self._rnn_states[env_ids] += 1
        return AgentStep(
            action = self._rnn_states[env_ids].copy(),
            agent_info = DummyAgentInfo(
                observation=observation,
                previous_action=previous_action,
                previous_reward=previous_reward,
            ),
        )

    def value(self, observation: Buffer, previous_action: Buffer, previous_reward: Buffer, *, env_ids: Union[int, slice]) -> Buffer:
        return AgentStep(
            action = self._rnn_states[env_ids].copy(),
            agent_info = DummyAgentInfo(
                observation=observation,
                previous_action=previous_action,
                previous_reward=previous_reward,
            ),
        )
