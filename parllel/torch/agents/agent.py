from typing import Iterable, Tuple, Union

import torch

from parllel.arrays import Array
from parllel.buffers import Buffer, buffer_method
from parllel.cages.collections import EnvSpaces
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

    def __init__(self, model: torch.nn.Module, distribution: Distribution,
            device: torch.device = None) -> AgentStep:
        self._model = model
        self._distribution = distribution

        # possibly move model to GPU
        if device is None:
            device = torch.device("cpu")
        if device != torch.device("cpu"):
            self.model.to(device)
        self._device = device

        self._mode = None
        self._rnn_states = None
        self._previous_action = None

    def reset(self) -> None:
        if self._rnn_states is not None:
            self._rnn_states[:] = 0
        if self._previous_action is not None:
            self._previous_action[:] = 0

    def reset_one(self, env_index) -> None:
        if self._rnn_states is not None:
            # rnn_states are of shape [N, B, H]
            self._rnn_states[:, env_index] = 0
        if self._previous_action is not None:
            self._previous_action[env_index] = 0

    def get_states(self, env_indices: Union[int, slice]):
        return (self._rnn_states[:, env_indices],
            self._previous_action[env_indices])

    def advance_states(self, next_rnn_states: Buffer, action: Buffer,
            env_indices: Union[int, slice]) -> Buffer[torch.Tensor]:
        if self._rnn_states is not None:
            # transpose the rnn_states from [N,B,H] -> [B,N,H] for storage.
            prev_rnn_state = buffer_method(self._rnn_states[:, env_indices],
                "transpose", 0, 1)

            # replace old rnn_states with new ones
            self._rnn_states[:, env_indices] = next_rnn_states
        else:
            prev_rnn_state = None
        
        if self._previous_action is not None:
            self._previous_action[env_indices] = action
        
        return prev_rnn_state

    def parameters(self) -> Iterable[torch.Tensor]:
        return self.model.parameters()

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
