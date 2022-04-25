from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from parllel.torch.distributions import MultiDistribution

from .agent import TorchAgent


@dataclass(frozen=True)
class AgentProfile:
    """A tuple describing all relevant information about a real agent. This
    allows multiple real agents to share a single agent instance and therefore
    share a single model.

    Args:
        instance: the agent instance containing the model
        obs_key: the part of the observation that this agent sees. None passes
            the whole observation.
        action_key: the part of action for which this agent is responsible.
    """
    instance: TorchAgent
    action_key: str
    obs_key: Optional[str] = None


class EnsembleAgent(TorchAgent):
    """Pseudo-agent that owns a collection of subagents defined by agent
    profiles. The subagents use a collection of neural networks, where some
    networks may be shared between agents. Each neural network is owned by an
    agent instance, which may be a multi-headed agent if the network is shared.
    Agent profiles describe which piece of the observation space the subagent
    observes and which piece of the action space the subagent is responsible
    for. If no models are shared between subagents, standard agent and
    model types may be used. If a model is shared between agents, multi-headed
    models and agents must be used.

    Args:
        agent_profiles: an agent profile defines a unique actor in the
            environment.
    """

    def __init__(self, agent_profiles: Sequence[AgentProfile]):

        self._agent_profiles = agent_profiles
        self._agent_instances = set(profile.instance for profile in self._agent_profiles)

        # allows convenient access to all parameters via ensemble agent's
        # model property
        model = torch.nn.ModuleDict({
            profile.action_key: profile.instance.model
            for profile
            in self._agent_profiles
        })

        # exposes multi-agent distribution methods to algorithm
        distribution = MultiDistribution({
            profile.action_key: profile.instance.distribution
            for profile
            in self._agent_profiles
        })

        devices = [profile.instance.device for profile in self._agent_profiles]
        device = devices[0]
        if not all(dev == device for dev in devices):
            raise ValueError("All agents must be on the same device.")

        super().__init__(model, distribution, device)

        self.recurrent = any(instance.recurrent for instance in self._agent_instances)

    def reset(self) -> None:
        for agent in self._agent_instances:
            agent.reset()

    def reset_one(self, env_index) -> None:
        for agent in self._agent_instances:
            agent.reset_one(env_index)

    def train_mode(self, elapsed_steps: int) -> None:
        self._mode = "train"
        for agent in self._agent_instances:
            agent.train_mode(elapsed_steps)

    def sample_mode(self, elapsed_steps: int) -> None:
        self._mode = "sample"
        for agent in self._agent_instances:
            agent.sample_mode(elapsed_steps)

    def eval_mode(self, elapsed_steps: int) -> None:
        self._mode = "eval"
        for agent in self._agent_instances:
            agent.eval_mode(elapsed_steps)
