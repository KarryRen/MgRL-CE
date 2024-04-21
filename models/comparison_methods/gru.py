# -*- coding: utf-8 -*-
# @Time    : 2024/4/6 15:15
# @Author  : Karry Ren

""" The Comparison Methods 1: GRU.

Ref. https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_gru.py#L294

"""

import logging
import torch
from torch import nn
from typing import Dict


class GRU_Net(nn.Module):
    """ The 2 Layer GRU. hidden_size=64. """

    def __init__(
            self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
            dropout: float = 0.0, device: torch.device = torch.device("cpu")
    ):
        """ The init function of GRU Net.

        :param input_size: input size for each time step
        :param hidden_size: hidden size of gru
        :param num_layers: the num of gru layers
        :param dropout: the dropout ratio
        :param device: the computing device

        """

        super(GRU_Net, self).__init__()
        self.device = device

        # ---- Log the info of GRU ---- #
        logging.info(f"|||| Using GRU Now ! input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout_ratio={dropout}||||")

        # ---- Part 1. The GRU module ---- #
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        ).to(device=device)

        # ---- Part 2. The output fully connect layer ---- #
        self.fc_out = nn.Linear(hidden_size, 1).to(device=device)

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

        # ---- Step 3. GRU Encoding and get the hidden_feature ---- #
        hidden_g1, _ = self.gru(feature_g1)  # shape=(bs, T, hidden_size)

        # ---- Step 4. FC to get the prediction ---- #
        # get the last step hidden feature of g1
        last_step_hidden_g1 = hidden_g1[:, -1, :]  # shape=(bs, hidden_size)
        # use the last step to predict
        y = self.fc_out(last_step_hidden_g1)  # shape=(bs, 1)

        # ---- Step 5. Return the output ---- #
        output = {"pred": y}
        return output


if __name__ == "__main__":  # A demo of GRU
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

    model = GRU_Net(input_size=1, device=dev)
    out = model(mg_input)
    print(out["pred"].shape)
