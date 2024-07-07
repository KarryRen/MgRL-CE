# -*- coding: utf-8 -*-
# @Time    : 2024/4/25 09:55
# @Author  : Karry Ren

""" The modules of ablation methods. """

from typing import Tuple
import torch
from torch import nn


class NoRec_FeatureEncoder(nn.Module):
    """ The basic block of feature encoder of each granularity for Mg_Add.

    Including 2 parts:
        - F_{TEnc}: Temporal Feature Encoder (2 Layer GRU), extracting the temporal feature `H`.
        - F_{Pred}: Prediction Net (2 Layer MLP), getting the prediction `y`.

    """

    def __init__(self, input_size: int, hidden_size: int):
        """ The init function of FeatureEncoder.

        :param input_size: the input size of encoding feature
        :param hidden_size: the hidden size of encoding feature

        """

        super(NoRec_FeatureEncoder, self).__init__()

        # ---- Part 1. F_{TEnc}: Temporal Feature Encoder ---- #
        self.temporal_feature_encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)

        # ---- Part 2. F_{Pred}: Prediction Net ---- #
        self.prediction_net = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        )

    def forward(self, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ The forward function of FeatureEncoder.

        :param P: the P of each granularity, shape=(bs, T, D*K), where D*K is the input_size
                  :math: {P_1 = F_1 && P_other(g) = F_other(g) - R_(g-1)}

        returns:
            - H: the hidden state of each granularity, shape=(bs, T, hidden_size)
            - y: the prediction of each granularity, shape=(bs, 1)

        """

        # ---- Step 1. Use the temporal_feature_encoder to encode P ---- #
        H, _ = self.temporal_feature_encoder(P)  # shape=(bs, T, hidden_size)

        # ---- Step 2. Use the prediction_net to get the prediction ---- #
        y = self.prediction_net(H[:, -1, :])  # shape=(bs, 1)

        # ---- Step 3. Return ---- #
        return H, y
