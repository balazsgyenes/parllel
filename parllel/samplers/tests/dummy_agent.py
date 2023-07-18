from typing import Union

import gymnasium as gym
import numpy as np
from numpy import random

from parllel.arrays import Array, buffer_from_dict_example
from parllel.buffers import (AgentSamples, Buffer, NamedTupleClass,
                             buffer_asarray, buffer_method)
from parllel.handlers import Agent, AgentStep
from parllel.types import BatchSpec

DummyInfo = NamedTupleClass("DummyAgentInfo", ["observation", "previous_state"])


class DummyAgent(Agent):
    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        batch_spec: BatchSpec,
        n_batches: int,
        recurrent: bool = False,
    ) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.batch_spec = batch_spec
        self.recurrent = recurrent

        self.states = Array(
            feature_shape=(),
            batch_shape=(n_batches * batch_spec.T, batch_spec.B),
            dtype=np.float32,
            padding=1,
        )

        self._step_ctr = 0
        batch_action = buffer_from_dict_example(
            self.action_space.sample(),
            batch_shape=(n_batches * batch_spec.T, batch_spec.B),
            name="action",
        )
        batch_info = buffer_from_dict_example(
            {
                "observation": self.observation_space.sample(),
                "previous_state": np.array(0, dtype=np.float32),
            },
            batch_shape=(n_batches * batch_spec.T, batch_spec.B),
            name="agentinfo",
        )
        self.samples = AgentSamples(batch_action, batch_info)
        self.values = Array(
            feature_shape=(),
            batch_shape=(n_batches, batch_spec.B),
            dtype=np.float32,
        )
        self.resets = Array(
            feature_shape=(),
            batch_shape=(n_batches * batch_spec.T, batch_spec.B),
            dtype=np.bool_,
            padding=1,
        )

        if self.recurrent:
            self.init_rnn_states = Array(
                feature_shape=(),
                batch_shape=(n_batches, batch_spec.B),
                dtype=np.float32,
            )

        self.rng = random.default_rng()

    def get_agent_info(self) -> DummyInfo:
        agent_info = buffer_asarray(self.samples.agent_info[0, 0])
        return buffer_method(agent_info, "copy")

    def reset(self) -> None:
        self.resets[self._step_ctr - 1] = True
        self.states[self._step_ctr] = 0

    def reset_one(self, env_index: int) -> None:
        if self.recurrent:
            # sampling batch may have stopped early
            batch_ctr = (self._step_ctr - 1) // self.batch_spec.T + 1
            # advance counter to the next batch
            self._step_ctr = batch_ctr * self.batch_spec.T
        self.resets[self._step_ctr - 1, env_index] = True
        self.states[self._step_ctr, env_index] = 0

    def initial_rnn_state(self) -> Buffer:
        # sampling batch may have stopped early
        batch_ctr = (self._step_ctr - 1) // self.batch_spec.T + 1
        # advance counter to the next batch
        self._step_ctr = batch_ctr * self.batch_spec.T

        init_rnn_state = self.rng.random(self.batch_spec.B)
        self.init_rnn_states[batch_ctr] = init_rnn_state
        return init_rnn_state

    def step(
        self,
        observation: Buffer,
        *,
        env_indices: Union[int, slice] = ...,
    ) -> AgentStep:
        action = self.action_space.sample()
        agent_info = DummyInfo(
            buffer_method(observation, "copy"),
            buffer_method(buffer_asarray(self.states[self._step_ctr]), "copy"),
        )
        self.samples.action[self._step_ctr] = action
        self.samples.agent_info[self._step_ctr] = agent_info
        next_state = self.rng.random(self.batch_spec.B)
        self.states[self._step_ctr + 1] = next_state
        self._step_ctr += 1

        return AgentStep(action, agent_info)

    def value(self, observation: Buffer) -> Buffer:
        batch_ctr = (self._step_ctr - 1) // self.batch_spec.T
        value = self.rng.random(self.batch_spec.B)
        self.values[batch_ctr] = value
        return value
