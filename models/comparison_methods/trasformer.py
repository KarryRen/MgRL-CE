# -*- coding: utf-8 -*-
# @Time    : 2024/4/15 23:26
# @Author  : Karry Ren

""" The Comparison Methods 3. Transformer.

Ref. https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_transformer.py#L258

"""

import logging
import torch
from torch import nn
from typing import Dict


class GRUNet(nn.Module):
    """ The 2 Layer GRU. hidden_size = 64. """

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

        super(GRUNet, self).__init__()
        self.device = device
        self.input_size = input_size

        # ---- Log the info of GRU ---- #
        logging.info(f"|||| Using GRU Now ! input_size={input_size}, hidden_size={hidden_size}, "
                     f"num_layers={num_layers}, dropout_ratio={dropout}||||")

        # ---- Part 1. The rnn of the GRU module ---- #
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        ).to(device=device)

        # ---- Part 2. The output fully connect layer ---- #
        self.fc_out = nn.Linear(hidden_size, 1).to(device=device)

    def forward(self, mul_granularity_input: Dict[str, torch.Tensor]):
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

        # ---- Step 1. Get the coarsest feature ---- #
        # g1 feature (coarsest), shape=(bs, T, K^g1, D)
        feature_g1 = mul_granularity_input["g1"].to(dtype=torch.float32, device=self.device)
        # get the feature shape
        bs, T = feature_g1.shape[0], feature_g1.shape[1]

        # ---- Step 2. Reshape the input for encoding ---- #
        feature_g1 = feature_g1.reshape(bs, T, self.input_size)  # shape=(bs, T, K^g1*D)

        # ---- Step 3. RNN Encoding and get the hidden_feature ---- #
        hidden_g1, _ = self.rnn(feature_g1)  # shape=(bs, T, hidden_size)

        # ---- Step 4. FC to get the prediction ---- #
        # get the last step hidden g1
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

    model = GRUNet(input_size=1, device=dev)
    out = model(mg_input)
    print(out["pred"].shape)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()
