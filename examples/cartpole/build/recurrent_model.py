from typing import List, Union

from gym import spaces
import torch
from torch import nn
import torch.nn.functional as F

from parllel.buffers import NamedArrayTupleClass
from parllel.torch.agents.categorical import ModelOutputs
from parllel.torch.models import MlpModel
from parllel.torch.utils import infer_leading_dims, restore_leading_dims


RnnState = NamedArrayTupleClass("RnnState", ["h", "c"])


class CartPoleLstmCategoricalPgModel(nn.Module):
    def __init__(self,
                 obs_space: spaces.Box,
                 action_space: spaces.Discrete,
                 pre_lstm_hidden_sizes: Union[int, List[int], None],
                 lstm_size: int,
                 post_lstm_hidden_sizes: Union[int, List[int], None],
                 hidden_nonlinearity: nn.Module,
                 ) -> None:
        super().__init__()
        assert isinstance(obs_space, spaces.Box)
        obs_shape = obs_space.shape[0]

        assert isinstance(action_space, spaces.Discrete)
        n_actions = action_space.n

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

    def forward(self, observation, previous_action, rnn_state=None):
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)

        fc_out = self.fc(observation.view(T * B, -1))

        lstm_input = [
            fc_out.view(T, B, -1),
            previous_action.view(T, B, -1),
        ]

        lstm_input = torch.cat(lstm_input, dim=-1)
        # convert namedarraytuple to tuple
        rnn_state = tuple(rnn_state) if rnn_state is not None else None
        lstm_out, (hn, cn) = self.lstm(lstm_input, rnn_state)

        output = self.head(lstm_out.view(T * B, -1))

        # form output values
        pi = F.softmax(output[..., :-1], dim=-1)
        value = output[..., -1]

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H]
        next_rnn_state = RnnState(h=hn, c=cn)

        return ModelOutputs(pi=pi, value=value, next_rnn_state=next_rnn_state)
