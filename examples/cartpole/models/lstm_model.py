from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor, nn

from parllel import ArrayDict
from parllel.torch.agents.categorical import DistParams, ModelOutputs
from parllel.torch.models import MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims


class CartPoleLstmPgModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Discrete,
        pre_lstm_hidden_sizes: Union[int, Sequence[int], None],
        lstm_size: int,
        post_lstm_hidden_sizes: Union[int, Sequence[int], None],
        hidden_nonlinearity: str,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        obs_shape = obs_space.shape[0]

        assert isinstance(action_space, spaces.Discrete)
        n_actions = action_space.n

        hidden_nonlinearity = getattr(nn, hidden_nonlinearity)

        self.fc = MlpModel(
            input_size=obs_shape,
            hidden_sizes=pre_lstm_hidden_sizes,
            output_size=None,
            hidden_nonlinearity=hidden_nonlinearity,
        )

        self.lstm = torch.nn.LSTM(
            input_size=self.fc.output_size + n_actions,
            hidden_size=lstm_size,
        )

        output_size = n_actions + 1
        self.head = MlpModel(
            input_size=lstm_size,
            hidden_sizes=post_lstm_hidden_sizes,
            output_size=output_size,
            hidden_nonlinearity=hidden_nonlinearity,
        )

    def forward(
        self,
        observation: Tensor,
        previous_action: Tensor,
        rnn_state: ArrayDict[Tensor] | None = None,
    ) -> ModelOutputs:
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)

        fc_out = self.fc(observation.view(T * B, -1))

        lstm_input = torch.cat(
            (
                fc_out.view(T, B, -1),
                previous_action.view(T, B, -1),
            ),
            dim=-1,
        )

        # convert array dict to tuple
        # TODO: verify correct order of rnn state values
        rnn_state = tuple(rnn_state.values()) if rnn_state is not None else None
        lstm_out, (hn, cn) = self.lstm(lstm_input, rnn_state)

        output = self.head(lstm_out.view(T * B, -1))

        # form output values
        probs = F.softmax(output[..., :-1], dim=-1)
        value = output[..., -1]

        # Restore leading dimensions: [T,B], [B], or [], as input.
        probs, value = restore_leading_dims((probs, value), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H]
        next_rnn_state = ArrayDict({"h": hn, "c": cn})

        return ModelOutputs(
            dist_params=DistParams(probs=probs),
            value=value,
            next_rnn_state=next_rnn_state,
        )
