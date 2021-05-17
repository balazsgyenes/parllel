
import torch

from parllel.torch.agents.agent import AgentStep, BaseAgent, AlternatingRecurrentAgentMixin
from parllel.agents.pg.base import (AgentInfo, ActorInfo, AgentInfoRnn, ActorInfoRnn,
    AgentOutputs, AgentOutputsRnn, ActorOutputs, ActorOutputsRnn)
from parllel.distributions.categorical import Categorical, DistInfo
from parllel.utils.buffer import buffer_to, buffer_func, buffer_method
from parllel.utils.collections import NamedArrayTupleSchema


ModelOutputs = NamedArrayTupleSchema("ModelOutputs", ["pi", "value"])
ModelOutputsRnn = NamedArrayTupleSchema(
    "ModelOutputsRnn", ["pi", "value", "next_rnn_state"])
ActorModelOutputs = NamedArrayTupleSchema("ModelOutputs", ["pi"])
ActorModelOutputsRnn = NamedArrayTupleSchema(
    "ModelOutputsRnn", ["pi", "next_rnn_state"])


class CategoricalPgAgent(BaseAgent):
    """
    Agent for policy gradient algorithm using categorical action distribution.
    Same as ``GausssianPgAgent`` and related classes, except uses
    ``Categorical`` distribution, and has a different interface to the model
    (model here outputs discrete probabilities in place of means and log_stds,
    while both output the value estimate).
    """

    def __init__(self, *args, recurrent=False, actor_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.recurrent = recurrent
        self.actor_only = actor_only
        if actor_only:
            # an actor cannot do value estimation
            self.InfoCls = ActorInfoRnn if recurrent else ActorInfo
            self.OutputCls = ActorOutputsRnn if recurrent else ActorOutputs

            def value_not_impl(self, observation, prev_action, prev_reward):
                raise NotImplementedError
            
            self.value = value_not_impl
        else:
            self.InfoCls = AgentInfoRnn if recurrent else AgentInfo
            self.OutputCls = AgentOutputsRnn if recurrent else AgentOutputs

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state=None):
        """Performs forward pass on training data, for algorithm."""
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = (observation, prev_action, prev_reward)
        if self.recurrent:
            model_inputs += (init_rnn_state,)
        model_inputs = buffer_to(model_inputs, device=self.device)
        model_outputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        output = (dist_info,)
        if not self.actor_only:
            output += (model_outputs.value,)
        output = buffer_to(output, device="cpu")
        if self.recurrent:  # Leave rnn_state on device
            output += (model_outputs.next_rnn_state,)
        return self.OutputCls(*output)

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B=global_B, env_ranks=env_ranks)
        self.distribution = Categorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to(
            (observation, prev_action, prev_reward), device=self.device
        )
        if self.recurrent:
            model_inputs += (self.prev_rnn_state,)  # already on device
        model_outputs = self.model(*model_inputs)
        dist_info = DistInfo(prob=model_outputs.pi)
        action = self.distribution.sample(dist_info)
        agent_info = (dist_info,)
        if not self.actor_only:
            agent_info += (model_outputs.value,)
        if self.recurrent:
            rnn_state = model_outputs.next_rnn_state
            # on first step after reset, prev_rnn_state is None
            # must ensure prev_rnn_state matches expected data type and size
            prev_rnn_state = self.prev_rnn_state or buffer_func(
                rnn_state, torch.zeros_like)
            # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
            prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
            agent_info += (prev_rnn_state,)
            # overwrite self.prev_rnn_state with new rnn_state
            self.advance_rnn_state(rnn_state)  # keep on device

        agent_info = self.InfoCls(*agent_info)
        agent_step = AgentStep(action=action, agent_info=agent_info)
        return buffer_to(agent_step, device="cpu")

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to(
            (observation, prev_action, prev_reward), device=self.device
        )
        if self.recurrent:
            model_inputs += (self.prev_rnn_state,)  # already on device
        model_outputs = self.model(*model_inputs)
        value = model_outputs.value
        return buffer_to(value, device="cpu")


class AlternatingRecurrentCategoricalPgAgent(AlternatingRecurrentAgentMixin,
        CategoricalPgAgent):
    pass
