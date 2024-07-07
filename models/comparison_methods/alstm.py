# -*- coding: utf-8 -*-
# @time   : 4/21/24 4:30 PM
# @Author  : Karry Ren

""" The Comparison Methods 7: ALSTM.

Ref. https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_alstm.py#L294

NOTE: Current version of implementation is just a simplified version of ALSTM. It is an LSTM with attention.

TODO: Will be updated 🔥.

"""

import logging
import torch
from torch import nn
from typing import Dict


class ALSTM_Net(nn.Module):
    """ The 2 Layer ALSTM. hidden_size=64. """

    def __init__(
            self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
            dropout: float = 0.0, device: torch.device = torch.device("cpu")
    ):
        """ The init function of ALSTM Net.

        :param input_size: input size for each time step
        :param hidden_size: hidden size of gru
        :param num_layers: the num of gru layers
        :param dropout: the dropout ratio
        :param device: the computing device

        """

        super(ALSTM_Net, self).__init__()
        self.device = device

        # ---- Log the info of ALSTM ---- #
        logging.info(f"|||| Using ALSTM Now ! input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout_ratio={dropout}||||")

        # ---- Part 1. Feature Encoding Net ---- #
        self.feature_mapping_net = nn.Sequential().to(device=device)
        self.feature_mapping_net.add_module("fc_in", nn.Linear(in_features=input_size, out_features=hidden_size).to(device=device))
        self.feature_mapping_net.add_module("act", nn.Tanh().to(device=device))

        # ---- Part 2. LSTM module ---- #
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout,
        ).to(device=device)

        # ---- Part 3. Attention Net ---- #
        self.attention_net = nn.Sequential().to(device=device)
        self.attention_net.add_module("att_fc_in", nn.Linear(in_features=hidden_size, out_features=int(hidden_size / 2)).to(device=device))
        self.attention_net.add_module("att_dropout", torch.nn.Dropout(dropout).to(device=device))
        self.attention_net.add_module("att_act", nn.Tanh().to(device=device))
        self.attention_net.add_module("att_fc_out", nn.Linear(in_features=int(hidden_size / 2), out_features=1, bias=False).to(device=device))
        self.attention_net.add_module("att_softmax", nn.Softmax(dim=1).to(device=device))

        # ---- Part 4. The output fully connect layer ---- #
        self.fc_out = nn.Linear(in_features=hidden_size * 2, out_features=1).to(device=device)

    def forward(self, mul_granularity_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ The forward function of GRU Net.

        :param mul_granularity_input: the input multi granularity, a dict with the format:
            {
                "g1": feature_g1,
                "g2": feature_g2,
                ...,
                "gG": feature_gG
            }

        returns: output, a dict with format:
            {"pred" : the prediction result, shape=(bs, 1)}

        """

        # ---- Step 1. Get the feature ---- #
        # g1 feature (coarsest), shape=(bs, T, K^g1, D)
        feature_g1 = mul_granularity_input["g1"].to(dtype=torch.float32, device=self.device)
        # get the feature shape
        bs, T, K_g1, d = feature_g1.shape[0], feature_g1.shape[1], feature_g1.shape[2], feature_g1.shape[3]

        # ---- Step 2. Preprocess the input for encoding ---- #
        feature_g1 = feature_g1.reshape(bs, T, K_g1 * d)  # reshape, shape=(bs, T, K^g1*D)

        # ---- Step 3. Encoding the feature ---- #
        mapping_feature_g1 = self.feature_mapping_net(feature_g1)  # shape=(bs, T, hidden_size)

        # ---- Step 4. Using the lstm to do the sequence encoding ---- #
        lstm_out_g1, _ = self.lstm(mapping_feature_g1)  # (bs, T, hidden_size)

        # ---- Step 5. Computing the attention score and weighting ---- #
        attention_score_g1 = self.attention_net(lstm_out_g1)  # shape=(bs, T, 1)
        lstm_out_att_g1 = torch.mul(lstm_out_g1, attention_score_g1)  # use the attention score to weight, shape=(bs, T, hidden_size)
        lstm_out_att_g1 = torch.sum(lstm_out_att_g1, dim=1)  # sum the hidden feature of each step, shape=(bs, 1, hidden_size)

        # ---- Step 6. FC to get the prediction ---- #
        # get the last step hidden g1
        last_step_hidden_g1 = lstm_out_g1[:, -1, :]  # shape=(bs, hidden_size)
        # cat the hidden state and attention feature and get the final prediction
        y = self.fc_out(torch.cat((last_step_hidden_g1, lstm_out_att_g1), dim=1))  # shape=(bs, 1)

        # ---- Step 7. Return the output ---- #
        output = {"pred": y}
        return output


if __name__ == "__main__":  # A demo of SFM
    bath_size, time_steps, D = 16, 4, 1
    mg_input = {
        "g1": torch.ones((bath_size, time_steps, 1, D)),
        "g2": torch.ones((bath_size, time_steps, 2, D)),
        "g3": torch.ones((bath_size, time_steps, 6, D)),
        "g4": torch.ones((bath_size, time_steps, 24, D)),
        "g5": torch.ones((bath_size, time_steps, 96, D))
    }
    g_dict = {"g1": 1, "g2": 2, "g3": 6, "g4": 24, "g5": 96}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ALSTM_Net(input_size=1, device=dev)
    out = model(mg_input)
    print(out["pred"].shape)
