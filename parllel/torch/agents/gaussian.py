from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor
from typing_extensions import NotRequired

from parllel import Array, ArrayDict, ArrayTree, Index, dict_map
from parllel.torch.distributions.gaussian import DistParams, Gaussian

from .pg import PgAgent, PgPrediction


class ModelOutputs(TypedDict):
    dist_params: DistParams
    value: NotRequired[Tensor]
    next_rnn_state: NotRequired[ArrayTree[Tensor]]


class GaussianPgAgent(PgAgent):
    """Agent for policy gradient algorithm using gaussian action distribution
    for continuous action spaces.

    The model must return the ModelOutputs type in this module, which contains:
        - mean and log_std: parameters defining a Gaussian distribution from
            which to sample actions for each state
        - value: value estimates, which can be omitted in cases without value
            prediction (e.g. vanilla PG) or where another entity predicts
            value (multi-agent scenarious)
        - next_rnn_state: the hidden recurrent state for the next time step

    The model must take between 1-3 arguments in the following order (arguments
    are only positional, not passed by keyword):
        - observation: current state of the environment
        - previous_action: action sampled from distribution from last time step
        - rnn_state: hidden recurrent state from last time step
    """

    distribution: Gaussian

    def __init__(
        self,
        model: torch.nn.Module,
        distribution: Gaussian,
        example_obs: ArrayTree[Array],
        example_action: ArrayTree[Array] | None = None,
        device: torch.device | None = None,
        recurrent: bool = False,
    ) -> None:
        super().__init__(model, distribution, device)

        self.recurrent = recurrent

        example_obs = example_obs.to_ndarray()
        example_obs = dict_map(torch.from_numpy, example_obs)
        example_obs = example_obs.to(device=self.device)
        example_inputs = (example_obs,)

        if self.recurrent:
            if example_action is None:
                raise ValueError(
                    "An example of an action is required for recurrent models."
                )
            example_action = example_action.to_ndarray()
            example_action = dict_map(torch.from_numpy, example_action)
            example_action = example_action.to(device=self.device)
            example_inputs += (example_action,)

        with torch.no_grad():
            try:
                # model will generate an rnn_state even if we don't pass one
                model_outputs: ModelOutputs = self.model(*example_inputs)
            except TypeError as e:
                raise TypeError(
                    "You may have forgotten to pass recurrent=True when creating this agent."
                ) from e

        if self.recurrent:
            assert "next_rnn_state" in model_outputs
            # store persistent agent state on device for next step
            self.rnn_states = model_outputs["next_rnn_state"]
            self.previous_action = example_action

    @torch.no_grad()
    def step(
        self,
        observation: ArrayTree[Array],
        *,
        env_indices: Index = ...,
    ) -> tuple[ArrayTree[Tensor], ArrayDict[Tensor]]:
        observation = observation.to_ndarray()
        observation = dict_map(torch.from_numpy, observation)
        observation = observation.to(device=self.device)
        model_inputs = (observation,)
        if self.recurrent:
            # already on device
            rnn_states, previous_action = self._get_states(env_indices)
            model_inputs += (previous_action, rnn_states)
        model_outputs: ModelOutputs = self.model(*model_inputs)

        # sample action from distribution returned by policy
        dist_params = model_outputs["dist_params"]
        action = self.distribution.sample(dist_params)

        # collect agent_info
        agent_info = ArrayDict({"dist_params": dist_params})
        if "value" in model_outputs:
            agent_info["value"] = model_outputs["value"]

        if self.recurrent:
            assert "next_rnn_state" in model_outputs
            # overwrite saved rnn_state and action as inputs to next step
            agent_info["previous_action"] = self._advance_states(
                model_outputs["next_rnn_state"],
                action,
                env_indices,
            )

        return action.cpu(), agent_info.cpu()

    @torch.no_grad()
    def value(self, observation: ArrayTree[Array]) -> ArrayTree[Tensor]:
        observation = observation.to_ndarray()
        observation = dict_map(torch.from_numpy, observation)
        observation = observation.to(device=self.device)
        model_inputs = (observation,)
        if self.recurrent:
            # already on device
            rnn_states, previous_action = self._get_states(...)
            model_inputs += (previous_action, rnn_states)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        assert "value" in model_outputs
        value = model_outputs["value"]
        return value.cpu()

    def predict(
        self,
        observation: ArrayTree[Tensor],
        agent_info: ArrayDict[Tensor],
        init_rnn_state: ArrayTree[Tensor] | None,
    ) -> PgPrediction:
        """Performs forward pass on training data, for algorithm."""
        model_inputs = (observation,)
        if self.recurrent:
            assert init_rnn_state is not None
            previous_action = agent_info["previous_action"]
            # rnn_states were saved into the samples buffer as [B,N,H]
            # transform back [B,N,H] --> [N,B,H].
            init_rnn_state = init_rnn_state.transpose(0, 1).contiguous()
            model_inputs += (previous_action, init_rnn_state)
        model_outputs: ModelOutputs = self.model(*model_inputs)
        return model_outputs
