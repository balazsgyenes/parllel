from typing import List, Optional, Union

from gym import spaces
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from parllel.buffers import NamedArrayTupleClass
from parllel.torch.agents.categorical import ModelOutputs
from parllel.torch.models import Conv2dHeadModel, MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims


RnnState = NamedArrayTupleClass("RnnState", ["h", "c"])


class AtariLstmPgModel(nn.Module):
    def __init__(
            self,
            obs_space: spaces.Box,
            action_space: spaces.Discrete,
            channels: List[int],
            kernel_sizes: List[int],
            strides: List[int],
            paddings: List[int],
            use_maxpool: bool,
            post_conv_hidden_sizes: Union[int, List[int], None],
            post_conv_output_size: Optional[int],
            post_conv_nonlinearity: str,
            lstm_size: int,
            post_lstm_hidden_sizes: Union[int, List[int], None],
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

    def forward(self, observation, previous_action, init_rnn_state=None):

        image = observation.type(torch.float)  # Requires torch.uint8 inputs
        image = image.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)

        # model params and inputs to the model have to share the same dtype -> cast to float32
        previous_action = previous_action.type(torch.float)

        encoding = self.conv(image.view(T * B, *img_shape))
        lstm_input = torch.cat([
            encoding.view(T, B, -1),
            previous_action.view(T, B, -1),
        ], dim=-1)

        # convert namedarraytuple to tuple
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)

        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        output = self.head(lstm_out.view(T * B, -1))

        pi = F.softmax(output[..., :-1], dim=-1)
        value = output[..., -1]

        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)
        next_rnn_state = RnnState(h=hn, c=cn)

        return ModelOutputs(pi = pi, value = value, next_rnn_state = next_rnn_state)
