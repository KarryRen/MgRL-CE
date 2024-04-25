# -*- coding: utf-8 -*-
# @Time    : 2024/4/15 23:26
# @Author  : Karry Ren

""" The Comparison Methods 3: Transformer.

Ref. https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_transformer.py#L258

"""

import logging
import torch
from torch import nn
from typing import Dict
import math


class PositionalEncoding(nn.Module):
    """ The Positional Encoding of Transformer. """

    def __init__(self, d_model: int, max_len: int = 1000):
        """ The init function of PositionalEncoding.

        :param d_model: the model dim
        :param max_len: the max position length

        """

        super(PositionalEncoding, self).__init__()

        # ---- Step 1. Construct the fix pe (all zero) ---- #
        pe = torch.zeros(max_len, d_model)

        # ---- Step 2. Computing the positional encoding data ---- #
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape=(max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # shape=(d_model // 2)
        pe[:, 0::2] = torch.sin(position * div_term)  # the even feature, shape=(max_len, d_model//w)
        pe[:, 1::2] = torch.cos(position * div_term)  # the odd feature, shape=(max_len, d_model//w)

        # ---- Step 3. Add the bs dim ---- #
        pe = pe.unsqueeze(0).transpose(0, 1)

        # ---- Step 4. Dump to the param ---- #
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ The forward function of PositionalEncoding.

        :param x: the feature need to do PE, shape=(T, bs, d_model)

        Attention: you should be careful about the feature dim, `T` is first !!!
        """
        return x + self.pe[:x.size(0), :]


class Transformer_Net(nn.Module):
    """ The 2 Layer Transformer. model dimension=64. """

    def __init__(
            self, d_feat: int, d_model: int = 64, n_head: int = 4, num_layers: int = 1,
            dropout: float = 0.0, device=torch.device("cpu")
    ):
        """ The init function of Transformer Net.

        :param d_feat: input dim of each step (input size)
        :param d_model: model dim (hidden size)
        :param n_head: the number of head for multi_head_attention
        :param dropout: the dropout ratio
        :param device: the computing device

        """

        super(Transformer_Net, self).__init__()
        self.device = device

        # ---- Log the info of Transformer ---- #
        logging.info(
            f"|||| Using Transformer Now ! d_feat={d_feat}, d_model={d_model}, "
            f"n_head={n_head}, num_layers={num_layers}, dropout_ratio={dropout}||||"
        )

        # ---- Part 1. Linear transformation layer ---- #
        self.linear_layer = nn.Linear(d_feat, d_model).to(device=device)

        # ---- Part 2. Positional Encoding ---- #
        self.pos_encoder = PositionalEncoding(d_model).to(device=device)

        # ---- Part 3. Transformer Encoder ---- #
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout).to(device=device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(device=device)

        # ---- Part 4. The output fully connect layer ---- #
        self.fc_out = nn.Linear(d_model, 1).to(device=device)

    def forward(self, mul_granularity_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ The forward function of Transformer Net.

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

        # ---- Step 3. Transformer Encoding ---- #
        # - step 3.1 transpose from (bs, T, K^g1*D) to (T, bs, K^g1*D)
        feature_g1 = feature_g1.transpose(1, 0)  # not batch first
        # - step 3.2 linear transformation
        feature_g1 = self.linear_layer(feature_g1)  # shape=(T, bs, d_model)
        # - step 3.3 positional encoding
        feature_g1 = self.pos_encoder(feature_g1)  # shape=(T, bs, d_model)
        # - step 3.4 transformer encoding
        trans_feature_g1 = self.transformer_encoder(feature_g1)  # shape=(T, bs, d_model)
        # - step 3.5 transpose back, from (T, bs, K^g1*D) to (bs, T, K^g1*D)
        trans_feature_g1 = trans_feature_g1.transpose(1, 0)  # batch first

        # ---- Step 4. FC to get the prediction ---- #
        # get the last step transformer feature of g1
        last_step_trans_feature_g1 = trans_feature_g1[:, -1, :]  # shape=(bs, hidden_size)
        # use the last step to predict
        y = self.fc_out(last_step_trans_feature_g1)  # shape=(bs, 1)

        # ---- Step 5. Return the output ---- #
        output = {"pred": y}
        return output


if __name__ == "__main__":  # A demo of Transformer
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

    model = Transformer_Net(d_feat=1, d_model=64, device=dev)
    out = model(mg_input)
    print(out["pred"].shape)
