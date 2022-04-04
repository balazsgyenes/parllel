from typing import Optional, Union

import gym
import numpy as np
from nptyping import NDArray

from parllel.buffers import Buffer, NamedTupleClass, buffer_method, buffer_map
from parllel.handlers import Agent, AgentStep


DummyAgentInfo = NamedTupleClass("DummyAgentInfo", ["observation", "state"])


class DummyAgent(Agent):
    def __init__(self, action_space: gym.Space, observation_space: gym.Space,
            recurrent: bool = False) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.recurrent = recurrent
    
    def dry_run(self, n_states: int, observation: Buffer):
        agent_info = (buffer_method(observation, "copy"),)
        if self.recurrent:
            self._n_states = n_states
            self._states = np.zeros(shape=(n_states,))
            state = self._states[0].copy()
            agent_info += (state,)
        agent_info = DummyAgentInfo(*agent_info)
        return agent_info, state
        
    def reset(self) -> None:
        self._states[:] = np.arange(self._n_states)
    
    def reset_one(self, env_index: int) -> None:
        self._states[env_index] = env_index

    def step(self, observation: Buffer, *, env_indices: Union[int, slice] = ...,
             ) -> AgentStep:
        agent_info = (buffer_method(observation, "copy"),)
        if self.recurrent:
            agent_info += (self._states[env_indices].copy(),)
            self._states[env_indices] += np.arange(self._n_states)[env_indices]
        return AgentStep(
            action = self._states[env_indices].copy() * 10,
            agent_info = DummyAgentInfo(*agent_info),
        )

    def value(self, observation: Buffer) -> Buffer:
        return self._states.copy() * 10
