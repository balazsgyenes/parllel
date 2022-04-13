from typing import Iterable, Union

import torch

from parllel.buffers import Buffer
from parllel.handlers import Agent
from parllel.torch.distributions import Distribution


class TorchAgent(Agent):
    """The agent manages a model and a sampling state for each environment
    instance. Outputs from the model are converted into actions during
    sampling, usually with the help of a distribution.
    """
    def __init__(self,
            model: torch.nn.Module,
            distribution: Distribution,
            device: torch.device = None
        ) -> None:
        self.model = model
        self.distribution = distribution

        # possibly move model to GPU
        if device is None:
            device = torch.device("cpu")
        if device != torch.device("cpu"):
            self.model.to(device)
        self.device = device

        self.mode = None
        self.recurrent = False
        self.rnn_states = None
        self.previous_action = None

    def reset(self) -> None:
        if self.rnn_states is not None:
            self.rnn_states[:] = 0
        if self.previous_action is not None:
            self.previous_action[:] = 0

    def reset_one(self, env_index) -> None:
        if self.rnn_states is not None:
            # rnn_states are of shape [N, B, H]
            self.rnn_states[:, env_index] = 0
        if self.previous_action is not None:
            self.previous_action[env_index] = 0

    def _get_states(self, env_indices: Union[int, slice]):
        # rnn_states has shape [N,B,H]
        rnn_state = self.rnn_states[:, env_indices]
        previous_action = self.previous_action[env_indices]
        return rnn_state, previous_action

    def _advance_states(self,
            next_rnn_states: Buffer,
            action: Buffer,
            env_indices: Union[int, slice]
        ) -> Buffer[torch.Tensor]:
        # rnn_states has shape [N,B,H]
        self.rnn_states[:, env_indices] = next_rnn_states
        
        # copy previous state before advancing, so that it it not overwritten
        previous_action = self.previous_action[env_indices].clone().detach()
        self.previous_action[env_indices] = action

        return previous_action

    def parameters(self) -> Iterable[torch.Tensor]:
        return self.model.parameters()

    def train_mode(self, elapsed_steps: int) -> None:
        """Go into training mode (e.g. see PyTorch's ``Module.train()``)."""
        self.model.train()
        self.mode = "train"

    def sample_mode(self, elapsed_steps: int) -> None:
        """Go into sampling mode."""
        self.model.eval()
        self.mode = "sample"

    def eval_mode(self, elapsed_steps: int) -> None:
        """Go into evaluation mode.  Example use could be to adjust epsilon-greedy."""
        self.model.eval()
        self.mode = "eval"
