# -*- coding: utf-8 -*-
# @Time    : 2024/4/2 15:32
# @Author  : Karry Ren

""" The modules of model, including 2 core modules:
        - FeatureEncoder: The basic block of feature encoder of each granularity for MgRLNet.
        - FeatureEncoderCE: The block of feature encoder with confidence estimating (CE) of each granularity for MgRL_CE_Net.

"""

from typing import Tuple
import torch
from torch import nn


class FeatureEncoder(nn.Module):
    """ The basic block of feature encoder of each granularity for MgRLNet.

    Including 3 parts:
        - F_{TEnc}: Temporal Feature Encoder (2 Layer GRU), extracting the temporal feature `H`.
        - F_{Pred}: Prediction Net (2 Layer MLP), getting the prediction `y`.
        - F_{Rec}: Reconstruction Net (2 Layer MLP), getting the reconstruction `R`.

    """

    def __init__(self, input_size: int, hidden_size: int):
        """ The init function of FeatureEncoder.

        :param input_size: the input size of encoding feature
        :param hidden_size: the hidden size of encoding feature

        """

        super(FeatureEncoder, self).__init__()

        # ---- Part 1. F_{TEnc}: Temporal Feature Encoder ---- #
        self.temporal_feature_encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)

        # ---- Part 2. F_{Pred}: Prediction Net ---- #
        self.prediction_net = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        )

        # ---- Part 3. F_{Rec}: Reconstruction Net ---- #
        self.reconstruction_net = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=input_size, bias=False)
        )

    def forward(self, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ The forward function of FeatureEncoder.

        :param P: the P of each granularity, shape=(bs, T, D*K), where D*K is the input_size
                  :math: {P_1 = F_1 && P_other(g) = F_other(g) - R_(g-1)}

        returns:
            - H: the hidden state of each granularity, shape=(bs, T, hidden_size)
            - y: the prediction of each granularity, shape=(bs, 1)
            - R: the reconstruction of each granularity, shape=(bs, T, D*K|input_size)

        """

        # ---- Step 1. Use the temporal_feature_encoder to encode P ---- #
        H, _ = self.temporal_feature_encoder(P)  # shape=(bs, T, hidden_size)

        # ---- Step 2. Use the prediction_net to get the prediction ---- #
        y = self.prediction_net(H[:, -1, :])  # shape=(bs, 1)

        # ---- Step 3. Use the reconstruction_net to get the reconstruction ---- #
        R = self.reconstruction_net(H)  # shape=(bs, T, D*K|input_size)

        # ---- Step 4. Return ---- #
        return H, y, R


class FeatureEncoderCE(nn.Module):
    """ The block of feature encoder with Confidence Estimating (CE) of each granularity for MgRL_CE_Net.

     Including 3 parts:
        - F_{TEnc}: Temporal Feature Encoder (2 Layer GRU), extracting the temporal feature `H`.
        - F_{Pred}: Prediction Net (2 Layer MLP), getting the prediction `y`
        - F_{Rec}: Reconstruction Net (2 Layer MLP), getting the reconstruction `R`

    """

    def __init__(self, input_size: int, hidden_size: int):
        """ The init function of FeatureEncoder.

        :param input_size: the input size of encoding feature
        :param hidden_size: the hidden size of encoding feature

        """

        super(FeatureEncoderCE, self).__init__()

        # ---- Part 1. F_{TEnc}: Temporal Feature Encoder ---- #
        self.temporal_feature_encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)

        # ---- Part 2. F_{Pred}: Prediction Net ---- #
        self.prediction_net = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        )

        # ---- Part 3. F_{Rec}: Reconstruction Net ---- #
        self.reconstruction_net = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=input_size, bias=False)
        )

    def forward(self, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ The forward function of FeatureEncoder.

        :param P: the P of each granularity, shape=(bs, T, D*K), where D*K is the encoding_input_size
            P_1 = F_1 && P_other(g) = F_other(g) - R_(g-1)

        returns:
            - H: the hidden state, shape=(bs, T, hidden_size)
            - y: the prediction, shape=(bs, 1)
            - R: the reconstruction, shape=(bs, T, D*K)

        """

        # ---- Step 1. Use the temporal_feature_encoder to encode P ---- #
        H, _ = self.temporal_feature_encoder(P)  # shape=(bs, T, hidden_size)

        # ---- Step 2. Use the prediction_net to get the prediction ---- #
        y = self.prediction_net(H[:, -1, :])  # shape=(bs, 1)

        # ---- Step 3. Use the reconstruction_net to get the reconstruction ---- #
        R = self.reconstruction_net(H)  # shape=(bs, T, D*K)

        # ---- Step 4. Return ---- #
        return H, y, R


if __name__ == '__main__':
    bs, T, input_size, hidden_size = 2, 5, 4, 8
    P1 = torch.randn((2, 5, 4))

    # ---- Test the feature encoder ---- #
    feature_encoder = FeatureEncoder(input_size=input_size, hidden_size=hidden_size)
    H, y, R = feature_encoder(P1)

    # ---- Test the feature encoder with Confidence Estimating (CE) ---- #

