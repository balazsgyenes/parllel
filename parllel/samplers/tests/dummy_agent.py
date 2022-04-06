from typing import Union

import gym
import numpy as np
from numpy import random
from parllel.arrays.utils import buffer_from_dict_example, buffer_from_example

from parllel.arrays import Array, buffer_from_dict_example
from parllel.buffers import Buffer, NamedTupleClass, buffer_method
from parllel.handlers import Agent, AgentStep
from parllel.buffers import AgentSamples
from parllel.types import BatchSpec


DummyAgentInfo = NamedTupleClass("DummyAgentInfo", ["observation"])


class DummyAgent(Agent):
    def __init__(self, action_space: gym.Space, observation_space: gym.Space,
            batch_spec: BatchSpec, n_batches: int, recurrent: bool = False) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.batch_B = batch_spec.B
        self.recurrent = recurrent

        self._step_ctr = 0
        self._batch_ctr = 0
        batch_action = buffer_from_dict_example(self.action_space.sample(),
            (n_batches * batch_spec.T, batch_spec.B), Array, name="action")
        batch_info = buffer_from_dict_example(
            {"observation": self.observation_space.sample()},
            (n_batches * batch_spec.T, batch_spec.B), Array, name="agentinfo")
        self._samples = AgentSamples(batch_action, batch_info)
        self._values = buffer_from_example(np.array(0, dtype=np.float32),
            (n_batches, batch_spec.B), Array)

        if self.recurrent:
            raise NotImplementedError

        self.rng = random.default_rng()

    def reset(self) -> None:
        if self.recurrent:
            self._states[:] = np.arange(self._n_states)
    
    def reset_one(self, env_index: int) -> None:
        if self.recurrent:
            self._states[env_index] = env_index

    def step(self, observation: Buffer, *, env_indices: Union[int, slice] = ...,
             ) -> AgentStep:
        action = self.action_space.sample()
        agent_info = DummyAgentInfo(buffer_method(observation, "copy"))
        self._samples.action[self._step_ctr] = action
        self._samples.agent_info[self._step_ctr] = agent_info
        self._step_ctr += 1

        # if self.recurrent:
        #     agent_info += (self._states[env_indices].copy(),)
        #     self._states[env_indices] += np.arange(self._n_states)[env_indices]

        return AgentStep(action, agent_info)

    def value(self, observation: Buffer) -> Buffer:
        value = self.rng.random(self.batch_B)
        self._values[self._batch_ctr] = value
        self._batch_ctr += 1
        return value

    @property
    def samples(self):
        return self._samples

    @property
    def values(self):
        return self._values
