from typing import Optional, Union

import gym
import numpy as np
from numpy import random
from nptyping import NDArray
from parllel.arrays.utils import buffer_from_dict_example, buffer_from_example

from parllel.arrays import Array, buffer_from_dict_example
from parllel.buffers import Buffer, NamedTupleClass, buffer_method, buffer_asarray
from parllel.handlers import Agent, AgentStep, Handler
from parllel.samplers import AgentSamples
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
        self._agent_samples = AgentSamples(batch_action, batch_info)
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
        self._agent_samples.action[self._step_ctr] = action
        self._agent_samples.agent_info[self._step_ctr] = agent_info
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
    def agent_samples(self):
        return self._agent_samples

    @property
    def values(self):
        return self._values


class DummyHandler(Handler):
    def step(self, observation: Buffer[Array], *, env_indices:
            Union[int, slice] = ..., out_action: Buffer[Array] = None,
            out_agent_info: Buffer[Array] = None) -> Optional[AgentStep]:
        
        observation: Buffer[NDArray] = buffer_asarray(observation)

        agent_step: AgentStep = self._agent.step(observation,
            env_indices=env_indices)

        if any(out is None for out in (out_action, out_agent_info)):
            return agent_step
        else:
            action, agent_info = agent_step
            out_action[:] = action
            out_agent_info[:] = agent_info

    def value(self, observation: Buffer[Array], *,
            out_value: Buffer[Array] = None) -> Optional[Buffer]:
        
        observation: Buffer[NDArray] = buffer_asarray(observation)

        value: Buffer[NDArray] = self._agent.value(observation)

        if out_value is None:
            return value
        else:
            out_value[:] = value
