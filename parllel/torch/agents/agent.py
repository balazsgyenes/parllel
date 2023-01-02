from os import PathLike
from typing import Callable, Optional, Union

import torch

from parllel.buffers import Buffer, buffer_map
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
            device: Optional[torch.device] = None,
        ) -> None:
        self.model = model
        self.distribution = distribution

        # possibly move model to GPU
        if device is None:
            device = torch.device("cpu")
        if device != torch.device("cpu"):
            self.model.to(device)
            self.distribution.to_device(device)
        self.device = device

        self.mode = "sample"
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
        try:
            # rnn_states has shape [N,B,H]
            rnn_state = self.rnn_states[:, env_indices]
        except TypeError as e:
            raise ValueError("Could not index into recurrent state. Make sure "
                "this agent is recurrent.") from e
        previous_action = self.previous_action[env_indices]
        return rnn_state, previous_action

    def _advance_states(self,
            next_rnn_states: Buffer,
            action: Buffer,
            env_indices: Union[int, slice]
        ) -> Buffer[torch.Tensor]:
        try:
            # rnn_states has shape [N,B,H]
            self.rnn_states[:, env_indices] = next_rnn_states
        except TypeError as e:
            raise ValueError("Could not index into recurrent state. Make sure "
                "this agent is recurrent.") from e
        
        # copy previous state before advancing, so that it it not overwritten
        previous_action = self.previous_action[env_indices].clone().detach()
        self.previous_action[env_indices] = action

        return previous_action

    def save_model(self, path: PathLike) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self,
        path: PathLike,
        load_to_device: Union[Callable, torch.device, str, None] = None,
    ) -> None:
        state_dict = torch.load(path, load_to_device)
        self.model.load_state_dict(state_dict)

    def train_mode(self, elapsed_steps: int) -> None:
        """Go into training mode (e.g. see PyTorch's ``Module.train()``)."""
        # does not set self.mode because there is no state associated with
        # training (i.e. we only care when moving between sample and eval
        # modes)
        self.model.train()

    def sample_mode(self, elapsed_steps: int) -> None:
        """Go into sampling mode."""
        self.model.eval()

        # if coming from evaluation, restore states from sampling
        if self.recurrent and self.mode == "eval":
            self.rnn_states = self._sampler_rnn_states
            self.previous_action = self._sampler_previous_action

        self.mode = "sample"

    def eval_mode(self, elapsed_steps: int) -> None:
        """Go into evaluation mode.  Example use could be to adjust epsilon-greedy."""
        self.model.eval()

        # if coming from sampling, store states and set new blank states
        if self.recurrent and self.mode == "sample":
            self._sampler_rnn_states = self.rnn_states
            self.rnn_states = buffer_map(torch.zeros_like, self.rnn_states)

            self._sampler_previous_action = self.previous_action
            self.previous_action = buffer_map(torch.zeros_like,
                self.previous_action)

        self.mode = "eval"
