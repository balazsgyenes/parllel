from typing import Iterable, Union

import torch

from parllel.buffers.buffer import Buffer
from parllel.handlers import Agent, AgentStep
from parllel.torch.distributions.base import Distribution


class TorchAgent(Agent):
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

    @property
    def device(self):
        return self._device

    def __init__(self, model: torch.nn.Module, distribution: Distribution, device: torch.device = None) -> AgentStep:
        self._model = model
        self._distribution = distribution

        # possibly move model to GPU
        if device is None:
            device = torch.device("cpu")
        if device != torch.device("cpu"):
            self.model.to(device)
        self._device = device

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

    def advance_rnn_states(self, next_rnn_states: Buffer, env_indices: Union[int, slice]):
        if self._rnn_states is not None:
            # rnn_states are of shape [N, B, H]
            self._rnn_states[:, env_indices] = next_rnn_states

    def parameters(self) -> Iterable[torch.Tensor]:
        """Parameters to be optimized (overwrite in subclass if multiple models)."""
        return self.model.parameters()

    def save_state_dict(self, filepath) -> None:
        """Returns model parameters for saving."""
        raise NotImplementedError

    def train_mode(self, elapsed_steps: int) -> None:
        """Go into training mode (e.g. see PyTorch's ``Module.train()``)."""
        self.model.train()
        self._mode = "train"

    def sample_mode(self, elapsed_steps: int) -> None:
        """Go into sampling mode."""
        self.model.eval()
        self._mode = "sample"

    def eval_mode(self, elapsed_steps: int) -> None:
        """Go into evaluation mode.  Example use could be to adjust epsilon-greedy."""
        self.model.eval()
        self._mode = "eval"
