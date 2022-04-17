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

        self.MultiAction = NamedArrayTupleClass(
            "MultiAction",
            [profile.action_key for profile in self._agent_profiles]
        )

        self.MultiDistInfo = NamedArrayTupleClass(
            "MultiDistInfo",
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
        actions, dist_infos, values, prev_actions = [], [], [], []
        for agent in self._agent_profiles:
            if agent.obs_key is not None:
                subagent_observation = getattr(observation, agent.obs_key)
            else:
                subagent_observation = observation

            subagent_action, subagent_info = agent.instance.step(
                subagent_observation, env_indices=env_indices)
            subagent_distinfo, subagent_value, subagent_prev_action = (
                subagent_info
            )

            actions.append(subagent_action)
            dist_infos.append(subagent_distinfo)
            values.append(subagent_value)
            prev_actions.append(subagent_prev_action)

        action = self.MultiAction(*actions)
        agent_info = AgentInfo(
            dist_info = self.MultiDistInfo(*dist_infos),
            # values must be array-like for subtraction of old_values from
            # current values in algo
            value = torch.stack(values, dim=-1),
            prev_action = self.MultiAction(*prev_actions)
        )
        return AgentStep(action, agent_info)

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

        return torch.stack(values, dim=-1)

    def predict(self, observation: Buffer, agent_info: AgentInfo,
                init_rnn_state: Optional[Buffer] = None,
                ) -> AgentPrediction:
        dist_infos = []
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

            dist_infos.append(subagent_dist_info)
            values.append(subagent_value)

        # dist_infos may have different dims, so cannot be stacked
        dist_info = self.MultiDistInfo(*dist_infos)
        # values must be array-like for subtraction of return_ from values
        # in algo
        value = torch.stack(values, dim=-1)
        return AgentPrediction(dist_info, value)
