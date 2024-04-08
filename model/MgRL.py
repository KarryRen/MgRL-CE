# -*- coding: utf-8 -*-
# @Time    : 2024/4/2 15:31
# @Author  : Karry Ren

""" The Multi-Granularity Residual Learning Framework (includes two core models):
        - The basic Multi-Granularity Residual Learning Net: MgRLNet.
        - Multi-granularity Residual Learning Framework with Confidence Estimation: MgRL_CE_Net.

"""

from typing import Dict
import torch
from torch import nn


class MgRLNet(nn.Module):
    """ The basic Multi-Granularity Residual Learning Net. """

    def __init__(
            self, granularity_dict: Dict[str, int], ga_K: int,
            encoding_input_size: int, encoding_hidden_size: int, device: torch.device
    ):
        """ The init function of MgRL Net.

        There are 2 main parts of Multi-Granularity Residual Learning Net:
            - Part 1. Granularity Alignment Module: Align the different granularity features to K dim
            - Part 2. Feature Encoding Module

        :param granularity_dict: the dict of input data granularity, should be format like:
            { "g1": g_1, "g2": g_2, ..., "gG":g_G}
        :param ga_K: the K of Granularity Alignment
        :param encoding_input_size: the input size of encoding feature
        :param encoding_hidden_size: the hidden size of encoding feature
        :param device: the computing device

        """

        super(MgRLNet, self).__init__()
        self.device = device

        # ---- Part 1. Granularity Alignment Module (Each Module includes 1 Linear Layer) ---- #
        self.granularity_alignment_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward wrong !
        # Granularity 1 (coarsest)
        self.granularity_alignment_dict["g1"] = nn.Linear(in_features=granularity_dict["g1"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 2
        self.granularity_alignment_dict["g2"] = nn.Linear(in_features=granularity_dict["g2"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 3
        self.granularity_alignment_dict["g3"] = nn.Linear(in_features=granularity_dict["g3"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 4
        self.granularity_alignment_dict["g4"] = nn.Linear(in_features=granularity_dict["g4"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 5 (finest)
        self.granularity_alignment_dict["g5"] = nn.Linear(in_features=granularity_dict["g5"], out_features=ga_K,
                                                          bias=False).to(device=device)

        # ---- Part 2. Feature Encoding Module (Each Module includes 3 Parts) ---- #
        self.feature_encoder_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward wrong !
        # Granularity 1 (coarsest)
        self.feature_encoder_dict["g1"] = nn.ModuleDict({})
        self.feature_encoder_dict["g1"]["Enc"] = nn.GRU(
            input_size=encoding_input_size, hidden_size=encoding_hidden_size, batch_first=True
        ).to(device=device)
        self.feature_encoder_dict["g1"]["Pred"] = nn.Sequential(
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=encoding_hidden_size, out_features=1, bias=False)
        ).to(device=device)
        self.feature_encoder_dict["g1"]["Rec"] = nn.Sequential(
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_input_size, bias=False)
        ).to(device=device)
        # Granularity 2
        self.feature_encoder_dict["g2"] = nn.ModuleDict({})
        self.feature_encoder_dict["g2"]["Enc"] = nn.GRU(
            input_size=encoding_input_size, hidden_size=encoding_hidden_size, batch_first=True
        ).to(device=device)
        self.feature_encoder_dict["g2"]["Pred"] = nn.Sequential(
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=encoding_hidden_size, out_features=1, bias=False)
        ).to(device=device)
        self.feature_encoder_dict["g2"]["Rec"] = nn.Sequential(
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_input_size, bias=False)
        ).to(device=device)
        # Granularity 3
        self.feature_encoder_dict["g3"] = nn.ModuleDict({})
        self.feature_encoder_dict["g3"]["Enc"] = nn.GRU(
            input_size=encoding_input_size, hidden_size=encoding_hidden_size, batch_first=True
        ).to(device=device)
        self.feature_encoder_dict["g3"]["Pred"] = nn.Sequential(
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=encoding_hidden_size, out_features=1, bias=False)
        ).to(device=device)
        self.feature_encoder_dict["g3"]["Rec"] = nn.Sequential(
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_input_size, bias=False)
        ).to(device=device)
        # Granularity 4
        self.feature_encoder_dict["g4"] = nn.ModuleDict({})
        self.feature_encoder_dict["g4"]["Enc"] = nn.GRU(
            input_size=encoding_input_size, hidden_size=encoding_hidden_size, batch_first=True
        ).to(device=device)
        self.feature_encoder_dict["g4"]["Pred"] = nn.Sequential(
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=encoding_hidden_size, out_features=1, bias=False)
        ).to(device=device)
        self.feature_encoder_dict["g4"]["Rec"] = nn.Sequential(
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_input_size, bias=False)
        ).to(device=device)
        # Granularity 5 (finest)
        self.feature_encoder_dict["g5"] = nn.ModuleDict({})
        self.feature_encoder_dict["g5"]["Enc"] = nn.GRU(
            input_size=encoding_input_size, hidden_size=encoding_hidden_size, batch_first=True
        ).to(device=device)
        self.feature_encoder_dict["g5"]["Pred"] = nn.Sequential(
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=encoding_hidden_size, out_features=1, bias=False)
        ).to(device=device)
        self.feature_encoder_dict["g5"]["Rec"] = nn.Sequential(
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(in_features=encoding_hidden_size, out_features=encoding_input_size, bias=False)
        ).to(device=device)

    def forward(self, mul_granularity_input: Dict[str, torch.Tensor]) -> dict:
        """ The forward function of MgRL Net.

        There are 2 main steps during forward:
            - Step 1. Align the granularity
            - Step 2. Encoding feature with the residual learning framework

        :param mul_granularity_input: the input multi granularity, a dict with the format:
            {
                "g1": feature_g1,
                "g2": feature_g2,
                ...,
                "gG": feature_gG
            }

        returns: output, a dict with format:
            {
                "pred" : the prediction result, shape=(bs, 1),
                "rec_residuals" : a tuple of reconstruction residual, each residual have the same shape=(bs, T, D*K)
            }

        """

        # ---- Step 1. Align the granularity ---- #
        # get the different granularity feature
        # - g1 feature (coarsest), shape=(bs, T, K^g1, D)
        feature_g1 = mul_granularity_input["g1"].to(dtype=torch.float32, device=self.device)
        # - g2 feature, shape=(bs, T, K^g2, D)
        feature_g2 = mul_granularity_input["g2"].to(dtype=torch.float32, device=self.device)
        # - g3 feature, shape=(bs, T, K^g3, D)
        feature_g3 = mul_granularity_input["g3"].to(dtype=torch.float32, device=self.device)
        # - g4 feature, shape=(bs, T, K^g4, D)
        feature_g4 = mul_granularity_input["g4"].to(dtype=torch.float32, device=self.device)
        # - g5 feature (finest), shape=(bs, T, K^g5, D)
        feature_g5 = mul_granularity_input["g5"].to(dtype=torch.float32, device=self.device)
        # transpose the feature for alignment
        feature_g1 = feature_g1.permute(0, 1, 3, 2)  # shape from (bs, T, K^g1, D) to (bs, T, D, K^g1)
        feature_g2 = feature_g2.permute(0, 1, 3, 2)  # shape from (bs, T, K^g2, D) to (bs, T, D, K^g2)
        feature_g3 = feature_g3.permute(0, 1, 3, 2)  # shape from (bs, T, K^g3, D) to (bs, T, D, K^g3)
        feature_g4 = feature_g4.permute(0, 1, 3, 2)  # shape from (bs, T, K^g4, D) to (bs, T, D, K^g4)
        feature_g5 = feature_g5.permute(0, 1, 3, 2)  # shape from (bs, T, K^g5, D) to (bs, T, D, K^g5)
        # get the shape
        bs, T = feature_g1.shape[0], feature_g1.shape[1]
        # align the feature with linear transform, get the `F`
        F_g1 = self.granularity_alignment_dict["g1"](feature_g1).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g2 = self.granularity_alignment_dict["g2"](feature_g2).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g3 = self.granularity_alignment_dict["g3"](feature_g3).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g4 = self.granularity_alignment_dict["g4"](feature_g4).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g5 = self.granularity_alignment_dict["g5"](feature_g5).reshape(bs, T, -1)  # shape=(bs, T, D*K)

        # ---- Step 2. Encoding Feature with the residual learning framework ---- #
        # - g1 feature encoding
        P_g1 = F_g1  # the P of granularity is F
        H_g1, _ = self.feature_encoder_dict["g1"]["Enc"](P_g1)  # shape=(bs, T, hidden_size)
        y_g1 = self.feature_encoder_dict["g1"]["Pred"](H_g1[:, -1, :])  # shape=(bs, 1)
        R_g1 = self.feature_encoder_dict["g1"]["Rec"](H_g1)  # shape=(bs, T, D*K)
        # - g2 feature encoding
        P_g2 = F_g2 - R_g1  # residual learning, shape=(bs, T, D*K)
        H_g2, _ = self.feature_encoder_dict["g2"]["Enc"](P_g2)  # shape=(bs, T, hidden_size)
        y_g2 = self.feature_encoder_dict["g2"]["Pred"](H_g2[:, -1, :])  # shape=(bs, 1)
        R_g2 = self.feature_encoder_dict["g2"]["Rec"](H_g2)  # shape=(bs, T, D*K)
        # - g3 feature encoding
        P_g3 = F_g3 - R_g2  # residual learning, shape=(bs, T, D*K)
        H_g3, _ = self.feature_encoder_dict["g3"]["Enc"](P_g3)  # shape=(bs, T, hidden_size)
        y_g3 = self.feature_encoder_dict["g3"]["Pred"](H_g3[:, -1, :])  # shape=(bs, 1)
        R_g3 = self.feature_encoder_dict["g3"]["Rec"](H_g3)  # shape=(bs, T, D*K)
        # - g4 feature encoding
        P_g4 = F_g4 - R_g3  # residual learning, shape=(bs, T, D*K)
        H_g4, _ = self.feature_encoder_dict["g4"]["Enc"](P_g4)  # shape=(bs, T, hidden_size)
        y_g4 = self.feature_encoder_dict["g4"]["Pred"](H_g4[:, -1, :])  # shape=(bs, 1)
        R_g4 = self.feature_encoder_dict["g4"]["Rec"](H_g4)  # shape=(bs, T, D*K)
        # - g5 feature encoding
        P_g5 = F_g5 - R_g4  # residual learning, shape=(bs, T, D*K)
        H_g5, _ = self.feature_encoder_dict["g5"]["Enc"](P_g5)  # shape=(bs, T, hidden_size)
        y_g5 = self.feature_encoder_dict["g5"]["Pred"](H_g5[:, -1, :])  # shape=(bs, 1)

        # ---- Step 3. Return ---- #
        output = {
            "pred": (y_g1 + y_g2 + y_g3 + y_g4 + y_g5) / 5,
            "rec_residuals": (P_g2, P_g3, P_g4, P_g5)
        }
        return output


if __name__ == '__main__':
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

    model = MgRLNet(granularity_dict=g_dict, ga_K=2, encoding_input_size=2, encoding_hidden_size=16, device=dev)
    out = model(mg_input)
    print(out["pred"].shape)
