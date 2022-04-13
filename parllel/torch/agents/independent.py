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

        # Create namedarraytuple to contain dist_info from each agent. We might
        # not be able to concatenate them if the agents have different action
        # spaces.
        # self.MultiDistInfoCls = NamedArrayTupleClass(
        #     "MultiDistInfoCls",
        #     list(profile.name for profile in self._agent_profiles)
        # )
        
        # if reward_keys are provided, name value estimate according to reward it
        # corresponds to. otherwise, value estimate names are not used
        if any(profile.reward_key is not None for profile in self._agent_profiles):
            self.MultiValue = NamedArrayTupleClass(
                "MultiValue",
                list(profile.reward_key for profile in self._agent_profiles)
            )
        else:
            self.MultiValue = NamedArrayTupleClass(
                "MultiValue",
                list(profile.name for profile in self._agent_profiles)
            )

        if any(profile.instance.recurrent for profile in self._agent_profiles):
            self.recurrent = True
            # Create namedarraytuple to contain rnn_state from each agent. We
            # might not be able to concatenate them if the agents have
            # different rnn sizes. Not all subagents are necessarily recurrent.
            self.MultiRnnState = NamedArrayTupleClass(
                "MultiRnnState",
                list(profile.name
                     for profile in self._agent_profiles
                     if profile.instance.recurrent)
            )
        else:
            self.recurrent = False

    @torch.no_grad()
    def step(self, observation: Buffer, *, env_indices: Union[int, slice] = ...,
             ) -> AgentStep:
        subagent_steps = {}
        for agent in self._agent_profiles:
            if agent.obs_key is not None:
                agent_observation = getattr(observation, agent.obs_key)
            else:
                agent_observation = observation

            subagent_steps[agent.action_key] = agent.instance.step(
                agent_observation, env_indices=env_indices)

        return collate_buffers(subagent_steps.values(),
                               subagent_steps.keys())

    @torch.no_grad()
    def initial_rnn_state(self) -> Buffer:
        subagent_rnn_states = []
        for agent in self._agent_profiles:
            if agent.instance.recurrent:
                subagent_rnn_states.append(agent.instance.initial_rnn_state())

        return self.MultiRnnState(*subagent_rnn_states)

    @torch.no_grad()
    def value(self, observation: Buffer) -> Buffer:
        values = []
        for agent in self._agent_profiles:
            if agent.obs_key is not None:
                agent_observation = getattr(observation, agent.obs_key)
            else:
                agent_observation = observation

            values.append(agent.instance.value(agent_observation))

        return self.MultiValue(*values)

    def predict(self, observation: Buffer, agent_info: AgentInfo,
                init_rnn_state: Optional[Buffer] = None,
                ) -> AgentPrediction:
        subagent_predictions = {}
        i = 0
        for agent in self._agent_profiles:
            if agent.obs_key is not None:
                agent_observation = getattr(observation, agent.obs_key)
            else:
                agent_observation = observation
            subagent_info = getattr(agent_info, agent.action_key)
            if agent.instance.recurrent:
                agent_init_rnn_state = init_rnn_state.get(i)
                i += 1
            else:
                agent_init_rnn_state = None

            subagent_predictions[agent.action_key] = agent.instance(
                agent_observation, subagent_info, agent_init_rnn_state)

        return collate_buffers(subagent_predictions.values(),
                               subagent_predictions.keys())
