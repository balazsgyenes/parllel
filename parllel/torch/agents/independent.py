from typing import Optional, Sequence, Union

import torch

from parllel.buffers import Buffer, NamedArrayTupleClass
from parllel.buffers.utils import collate_buffers
from parllel.handlers.agent import AgentStep

from .ensemble import EnsembleAgent, AgentProfile
from .pg import AgentInfo, AgentPrediction


class IndependentPgAgents(EnsembleAgent):

    def __init__(self, agent_profiles: Sequence[AgentProfile]):
        super().__init__(agent_profiles)

        self.MultiValue = NamedArrayTupleClass(
            "MultiValue",
            [profile.action_key for profile in self._agent_profiles]
        )

        if any(profile.instance.recurrent for profile in self._agent_profiles):
            self.recurrent = True
            # Create namedarraytuple to contain rnn_state from each agent. We
            # might not be able to concatenate them if the agents have
            # different rnn sizes. Not all subagents are necessarily recurrent.
            self.MultiRnnState = NamedArrayTupleClass(
                "MultiRnnState",
                [profile.action_key for profile in self._agent_profiles]
            )
        else:
            self.recurrent = False

    @torch.no_grad()
    def step(self, observation: Buffer, *, env_indices: Union[int, slice] = ...,
             ) -> AgentStep:
        subagent_steps = {}
        for agent in self._agent_profiles:
            if agent.obs_key is not None:
                subagent_observation = getattr(observation, agent.obs_key)
            else:
                subagent_observation = observation

            subagent_steps[agent.action_key] = agent.instance.step(
                subagent_observation, env_indices=env_indices)

        return collate_buffers(subagent_steps.values(),
                               subagent_steps.keys())

    @torch.no_grad()
    def initial_rnn_state(self) -> Buffer:
        subagent_rnn_states = [
            (
                agent.instance.initial_rnn_state()
                if agent.instance.recurrent
                else None
            )
            for agent in self._agent_profiles
        ]
        return self.MultiRnnState(*subagent_rnn_states)

    @torch.no_grad()
    def value(self, observation: Buffer) -> Buffer:
        values = []
        for agent in self._agent_profiles:
            if agent.obs_key is not None:
                subagent_observation = getattr(observation, agent.obs_key)
            else:
                subagent_observation = observation

            values.append(agent.instance.value(subagent_observation))

        return self.MultiValue(*values)

    def predict(self, observation: Buffer, agent_info: AgentInfo,
                init_rnn_state: Optional[Buffer] = None,
                ) -> AgentPrediction:
        dist_infos = {}
        values = []
        for agent in self._agent_profiles:
            if agent.obs_key is not None:
                subagent_observation = getattr(observation, agent.obs_key)
            else:
                subagent_observation = observation
            
            subagent_info = getattr(agent_info, agent.action_key)
            subagent_rnn_state = getattr(init_rnn_state, agent.action_key)

            subagent_dist_info, subagent_value = agent.instance(
                subagent_observation, subagent_info, subagent_rnn_state)

            dist_infos[agent.action_key] = subagent_dist_info
            values.append(subagent_value)

        # dist_infos are NamedTuples, so they can be collated
        dist_info = collate_buffers(dist_infos.values(), dist_infos.keys())
        # values must be stacked so they can be multiplied by advantage
        value = torch.stack(values, dim=-1)
        return AgentPrediction(dist_info, value)
