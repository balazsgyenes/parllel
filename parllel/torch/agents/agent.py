from dataclasses import dataclass
from typing import Any, Iterable, Union
from nptyping import NDArray

import torch


from parllel.torch.distributions.distribution import Distribution


@dataclass(frozen=True)
class AgentStep:
    action: Any
    agent_info: Any


class Agent:
    """The agent manages as a model and a sampling state for each environment
    instance. Outputs from the model are converted into actions during
    sampling, usually with the help of a distribution.
    """
    @property
    def model(self):
        return self._model

    @property
    def distribution(self):
        return self._distribution

    def initialize(self, model: torch.Module, device: torch.device, distribution: Distribution) -> AgentStep:
        self._model = model
        self.device = device
        self._distribution = distribution
        self._mode = None
        # subclass must override and create whatever state it needs
        self._rnn_states = None

    def reset(self) -> None:
        if self._rnn_states is not None:
            self._rnn_states[:] = 0

    def reset_one(self, env_index) -> None:
        if self._rnn_states is not None:
            # rnn_states are of shape [N, B, H]
            self._rnn_states[:, env_index] = 0

    @torch.no_grad()
    def step(self, observation: NDArray, prev_action: NDArray, prev_reward: NDArray, env_ids: Union[int, slice]) -> AgentStep:
        """Returns selected actions for environment instances in sampler."""
        raise NotImplementedError

    def evaluate(self, observation: NDArray, prev_action: NDArray, prev_reward: NDArray, prev_rnn_state: torch.Tensor = None) -> Any:
        """Returns values from model forward pass on training data (i.e. used
        in algorithm)."""
        raise NotImplementedError

    def parameters(self) -> Iterable[torch.Tensor]:
        """Parameters to be optimized (overwrite in subclass if multiple models)."""
        return self.model.parameters()

    def save_state_dict(self, filepath) -> None:
        """Returns model parameters for saving."""
        raise NotImplementedError

    def train_mode(self, steps_elapsed: int) -> None:
        """Go into training mode (e.g. see PyTorch's ``Module.train()``)."""
        self.model.train()
        self._mode = "train"

    def sample_mode(self, steps_elapsed: int) -> None:
        """Go into sampling mode."""
        self.model.eval()
        self._mode = "sample"

    def eval_mode(self, steps_elapsed: int) -> None:
        """Go into evaluation mode.  Example use could be to adjust epsilon-greedy."""
        self.model.eval()
        self._mode = "eval"
