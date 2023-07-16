from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor, nn

from parllel import ArrayDict
from parllel.torch.agents.categorical import ModelOutputs
from parllel.torch.models import Conv2dHeadModel, MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims


class AtariLstmPgModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Discrete,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int],
        use_maxpool: bool,
        post_conv_hidden_sizes: Union[int, Sequence[int], None],
        post_conv_output_size: Optional[int],
        post_conv_nonlinearity: str,
        lstm_size: int,
        post_lstm_hidden_sizes: Union[int, Sequence[int], None],
        post_lstm_nonlinearity: str,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        assert obs_space.dtype == np.uint8
        image_shape = tuple(obs_space.shape)

        assert isinstance(action_space, spaces.Discrete)
        n_actions = action_space.n

        post_conv_nonlinearity = getattr(nn, post_conv_nonlinearity)
        post_lstm_nonlinearity = getattr(nn, post_lstm_nonlinearity)

        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_maxpool=use_maxpool,
            hidden_sizes=post_conv_hidden_sizes,
            output_size=post_conv_output_size,
            nonlinearity=post_conv_nonlinearity,
        )

        self.lstm = torch.nn.LSTM(
            input_size=self.conv.output_size + n_actions,
            hidden_size=lstm_size,
        )

        self.head = MlpModel(
            input_size=lstm_size,
            hidden_sizes=post_lstm_hidden_sizes,
            output_size=n_actions + 1,
            hidden_nonlinearity=post_lstm_nonlinearity,
        )

    def forward(
        self,
        observation: Tensor,
        previous_action: Tensor,
        rnn_state: ArrayDict[Tensor] | None = None,
    ) -> ModelOutputs:
        image = observation.type(torch.float)  # Requires torch.uint8 inputs
        image = image.mul_(1.0 / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)

        # model params and inputs to the model have to share the same dtype -> cast to float32
        previous_action = previous_action.type(torch.float)

        encoding = self.conv(image.view(T * B, *img_shape))
        lstm_input = torch.cat(
            (
                encoding.view(T, B, -1),
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
        prob = F.softmax(output[..., :-1], dim=-1)
        value = output[..., -1]

        # Restore leading dimensions: [T,B], [B], or [], as input.
        prob, value = restore_leading_dims((prob, value), lead_dim, T, B)
        next_rnn_state = ArrayDict({"h": hn, "c": cn})

        return ModelOutputs(prob=prob, value=value, next_rnn_state=next_rnn_state)
