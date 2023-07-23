from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from gymnasium import spaces
import torch
from torch import Tensor

from parllel import Array, ArrayDict, ArrayTree, Index

from .ensemble import EnsembleAgent
from .pg import PgAgent, PgPrediction


@dataclass(frozen=True)
class PgAgentProfile:
    """A tuple describing all relevant information about a real agent. This
    allows multiple real agents to share a single agent instance and therefore
    share a single model.

    Args:
        instance: the agent instance containing the model
        obs_key: the part of the observation that this agent sees. None passes
            the whole observation.
        action_key: the part of action for which this agent is responsible.
    """

    instance: PgAgent
    action_key: str
    obs_key: str | None = None


class IndependentPgAgents(EnsembleAgent, PgAgent):
    _agent_profiles: Sequence[PgAgentProfile]
    _agent_instances: Sequence[PgAgentProfile]

    # TODO: add support for arbitrary agent_info

    def __init__(
        self,
        agent_profiles: Sequence[PgAgentProfile],
        action_space: spaces.Dict,
    ) -> None:
        # order agent profiles according to elements in action space
        # this ensures that the elements of all NamedTuples are in the same
        # order as defined in the environment's action space
        agent_profiles_ordered = []
        action_keys = [agent.action_key for agent in agent_profiles]
        for key in action_space:
            try:
                i = action_keys.index(key)
            except ValueError:
                raise ValueError(f"No agent is responsible for action {key}")

            agent_profiles_ordered.append(agent_profiles[i])

        super().__init__(agent_profiles_ordered)

    @torch.no_grad()
    def step(
        self,
        observation: ArrayTree[Array],
        *,
        env_indices: Index = ...,
    ) -> tuple[ArrayDict[Tensor], ArrayDict[Tensor]]:
        action, dist_params, prev_action = ArrayDict(), {}, {}
        values = []
        for agent in self._agent_profiles:
            if agent.obs_key is not None:
                assert isinstance(observation, Mapping)
                subagent_observation = observation[agent.obs_key]
            else:
                subagent_observation = observation

            action[agent.action_key], subagent_info = agent.instance.step(
                subagent_observation, env_indices=env_indices
            )
            dist_params[agent.action_key] = subagent_info["dist_params"]
            values.append(subagent_info["value"])
            if agent.instance.recurrent:
                prev_action[agent.action_key] = subagent_info["previous_action"]

        agent_info = ArrayDict(
            {
                "dist_params": dist_params,
                # values must be array-like for subtraction of old_values from
                # current values in algo
                "value": torch.stack(values, dim=-1),
            }
        )
        if prev_action:
            agent_info["previous_action"] = prev_action
        return action, agent_info

    @torch.no_grad()
    def initial_rnn_state(self) -> ArrayTree[Tensor]:
        rnn_state = ArrayDict(
            {
                agent.action_key: agent.instance.initial_rnn_state()
                for agent in self._agent_profiles
                if agent.instance.recurrent
            }
        )
        return rnn_state

    @torch.no_grad()
    def value(self, observation: ArrayTree[Array]) -> Tensor:
        values = []
        for agent in self._agent_profiles:
            if agent.obs_key is not None:
                assert isinstance(observation, Mapping)
                subagent_observation = observation[agent.obs_key]
            else:
                subagent_observation = observation

            values.append(agent.instance.value(subagent_observation))

        return torch.stack(values, dim=-1)

    def predict(
        self,
        observation: ArrayTree[Tensor],
        agent_info: ArrayDict[Tensor],
        init_rnn_state: ArrayDict[Tensor] | None,
    ) -> PgPrediction:
        previous_actions: ArrayDict[Tensor] = agent_info.get("previous_action")
        dist_params = {}
        values = []
        for agent in self._agent_profiles:
            if agent.obs_key is not None:
                assert isinstance(observation, Mapping)
                subagent_observation = observation[agent.obs_key]
            else:
                subagent_observation = observation

            subagent_info = ArrayDict()
            subagent_rnn_state = None
            if agent.instance.recurrent:
                assert previous_actions is not None
                subagent_info["previous_action"] = previous_actions[agent.action_key]
                assert init_rnn_state is not None
                subagent_rnn_state = init_rnn_state[agent.action_key]

            subagent_prediction = agent.instance.predict(
                subagent_observation,
                subagent_info,
                subagent_rnn_state,
            )

            dist_params[agent.action_key] = subagent_prediction["dist_params"]
            values.append(subagent_prediction["value"])

        # values must be array-like for subtraction of return_ from values
        # in algo
        value = torch.stack(values, dim=-1)
        return PgPrediction(dist_params=dist_params, value=value)
