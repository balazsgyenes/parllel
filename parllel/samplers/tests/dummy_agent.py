from typing import Union

import gymnasium as gym
import numpy as np
from numpy import random

from parllel import Array, ArrayTree, Index, dict_map
from parllel.agents import Agent
from parllel.tree.array_dict import ArrayDict
from parllel.types import BatchSpec


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
        self.samples = ArrayDict()
        self.samples["action"] = dict_map(
            Array.from_numpy,
            self.action_space.sample(),
            batch_shape=(n_batches * batch_spec.T, batch_spec.B),
        )
        self.samples["agent_info"] = dict_map(
            Array.from_numpy,
            {
                "observation": self.observation_space.sample(),
                "previous_state": np.array(0, dtype=np.float32),
            },
            batch_shape=(n_batches * batch_spec.T, batch_spec.B),
        )
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
            self.initial_rnn_states = Array(
                feature_shape=(),
                batch_shape=(n_batches, batch_spec.B),
                dtype=np.float32,
            )

        self.rng = random.default_rng()

    def get_agent_info(self) -> ArrayDict[np.ndarray]:
        return self.samples["agent_info"][0, 0].to_ndarray().copy()

    def reset(self) -> None:
        self.resets[self._step_ctr - 1] = True
        self.states[self._step_ctr] = 0

    def reset_one(self, env_index: int) -> None:
        if self.recurrent:
            # sampling batch may have stopped early
            batch_ctr = (self._step_ctr - 1) // self.batch_spec.T + 1
            # advance counter to the next batch
            self._step_ctr = batch_ctr * self.batch_spec.T
        if isinstance(env_index, list):
            env_index = np.asarray(env_index, dtype=np.int64)
        if isinstance(env_index, np.ndarray) and env_index.dtype == bool:
            index_tup = env_index.nonzero()
            assert len(index_tup) == 1
            env_index = index_tup[0]
        self.resets[self._step_ctr - 1, env_index] = True
        self.states[self._step_ctr, env_index] = 0

    def initial_rnn_state(self) -> np.ndarray:
        # sampling batch may have stopped early
        batch_ctr = (self._step_ctr - 1) // self.batch_spec.T + 1
        # advance counter to the next batch
        self._step_ctr = batch_ctr * self.batch_spec.T

        initial_rnn_state = self.rng.random(self.batch_spec.B)
        self.initial_rnn_states[batch_ctr] = initial_rnn_state
        return initial_rnn_state

    def step(
        self,
        observation: ArrayTree[Array],
        *,
        env_indices: Index = ...,
    ) -> tuple[ArrayTree, ArrayDict]:
        # write values into internal arrays
        for b in range(self.batch_spec.B):
            self.samples["action"][self._step_ctr, b] = self.action_space.sample()
        self.samples["agent_info"][self._step_ctr] = {
            "observation": observation,
            "previous_state": self.states[self._step_ctr],
        }

        # prepare return values by making a copy of internal arrays
        sample = self.samples[self._step_ctr].to_ndarray().copy()
        action = sample["action"]
        agent_info = sample["agent_info"]

        # update states and counter
        next_state = self.rng.random(self.batch_spec.B)
        self.states[self._step_ctr + 1] = next_state
        self._step_ctr += 1

        return action, agent_info

    def value(self, observation: ArrayTree[Array]) -> np.ndarray:
        batch_ctr = (self._step_ctr - 1) // self.batch_spec.T
        value = self.rng.random(self.batch_spec.B)
        self.values[batch_ctr] = value
        return value
