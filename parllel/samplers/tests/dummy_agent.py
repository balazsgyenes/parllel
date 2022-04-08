from typing import Union

import gym
import numpy as np
from numpy import random

from parllel.arrays import Array, RotatingArray, buffer_from_dict_example, buffer_from_example
from parllel.buffers import Buffer, NamedTupleClass, buffer_asarray, buffer_method
from parllel.handlers import Agent, AgentStep
from parllel.buffers import AgentSamples
from parllel.types import BatchSpec


DummyInfo = NamedTupleClass("DummyAgentInfo", ["observation", "previous_state"])

class DummyAgent(Agent):
    def __init__(self, action_space: gym.Space, observation_space: gym.Space,
            batch_spec: BatchSpec, n_batches: int, recurrent: bool = False) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.batch_spec = batch_spec
        self.recurrent = recurrent

        self._states = buffer_from_example(np.array(0, dtype=np.float32),
            (n_batches * batch_spec.T, batch_spec.B), RotatingArray)

        self._step_ctr = 0
        batch_action = buffer_from_dict_example(self.action_space.sample(),
            (n_batches * batch_spec.T, batch_spec.B), Array, name="action")
        batch_info = buffer_from_dict_example(
            {
                "observation": self.observation_space.sample(),
                "previous_state": np.array(0, dtype=np.float32),
            },
            (n_batches * batch_spec.T, batch_spec.B), Array, name="agentinfo")
        self._samples = AgentSamples(batch_action, batch_info)
        self._values = buffer_from_example(np.array(0, dtype=np.float32),
            (n_batches, batch_spec.B), Array)
        self._batch_resets = buffer_from_example(True,
            (n_batches * batch_spec.T, batch_spec.B), RotatingArray, padding=1)

        if self.recurrent:
            self._init_rnn_states = buffer_from_example(np.array(0, dtype=np.float32),
            (n_batches, batch_spec.B), Array)

        self.rng = random.default_rng()

    def dry_run(self) -> DummyInfo:
        agent_info = buffer_asarray(self._samples.agent_info[0, 0])
        return buffer_method(agent_info, "copy")

    def reset(self) -> None:
        self._batch_resets[self._step_ctr - 1] = True
        self._states[self._step_ctr] = 0
    
    def reset_one(self, env_index: int) -> None:
        if self.recurrent:
            # sampling batch may have stopped early
            batch_ctr = (self._step_ctr - 1) // self.batch_spec.T + 1
            # advance counter to the next batch
            self._step_ctr = batch_ctr * self.batch_spec.T
        self._batch_resets[self._step_ctr - 1, env_index] = True
        self._states[self._step_ctr, env_index] = 0

    def initial_rnn_state(self) -> Buffer:
        # sampling batch may have stopped early
        batch_ctr = (self._step_ctr - 1) // self.batch_spec.T + 1
        # advance counter to the next batch
        self._step_ctr = batch_ctr * self.batch_spec.T
        
        init_rnn_state = self.rng.random(self.batch_spec.B)
        self._init_rnn_states[batch_ctr] = init_rnn_state
        return init_rnn_state

    def step(self, observation: Buffer, *, env_indices: Union[int, slice] = ...,
             ) -> AgentStep:
        action = self.action_space.sample()
        agent_info = DummyInfo(
            buffer_method(observation, "copy"),
            buffer_method(buffer_asarray(self._states[self._step_ctr]), "copy"),
        )
        self._samples.action[self._step_ctr] = action
        self._samples.agent_info[self._step_ctr] = agent_info
        next_state = self.rng.random(self.batch_spec.B)
        self._states[self._step_ctr + 1] = next_state
        self._step_ctr += 1

        return AgentStep(action, agent_info)

    def value(self, observation: Buffer) -> Buffer:
        batch_ctr = (self._step_ctr - 1) // self.batch_spec.T
        value = self.rng.random(self.batch_spec.B)
        self._values[batch_ctr] = value
        return value

    @property
    def samples(self):
        return self._samples

    @property
    def values(self):
        return self._values

    @property
    def resets(self):
        return self._batch_resets

    @property
    def init_rnn_states(self):
        return self._init_rnn_states

    @property
    def states(self):
        return self._states
