# -*- coding: utf-8 -*-
# @Time    : 2024/4/2 15:31
# @Author  : Karry Ren

""" Two models:
    - The basic Multi-Granularity Residual Learning Net: MgRLNet.
    -
    - The Multi-granularity Residual Learning Framework with Confidence Estimation: MgRL_CE_Net.

"""

from typing import Dict, List
import torch
from torch import nn
import torch.nn.functional as F

from models.modules import FeatureEncoder, FeatureEncoderCE, NoPred_FeatureEncoder
from models.modules import ScaledDotProductAttention


class MgRL_Net(nn.Module):
    """ The basic Multi-Granularity Residual Learning Net. """

    def __init__(
            self, granularity_dict: Dict[str, int], ga_K: int,
            encoding_input_size: int, encoding_hidden_size: int, device: torch.device
    ):
        """ The init function of MgRL Net.

        There are 2 main parts of Multi-Granularity Residual Learning Net:
            - Part 1. Granularity Alignment Module (granularity_alignment): Align the different granularity features to K dim
            - Part 2. Feature Encoding Module (feature_encoder): Encoding the K dim feature and get the 3 outputs

        :param granularity_dict: the dict of input data granularity, should be format like: { "g1": g_1, "g2": g_2, ..., "gG":g_G}
        :param ga_K: the K of Granularity Alignment
        :param encoding_input_size: the input size of encoding feature
        :param encoding_hidden_size: the hidden size of encoding feature
        :param device: the computing device

        """

        super(MgRL_Net, self).__init__()
        self.device = device

        # ---- Part 1. Granularity Alignment Module (Each Module includes 1 Linear Layer) ---- #
        self.granularity_alignment_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward wrong !
        # Granularity 1 (coarsest)
        self.granularity_alignment_dict["g1"] = nn.Linear(in_features=granularity_dict["g1"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 2 (fine)
        self.granularity_alignment_dict["g2"] = nn.Linear(in_features=granularity_dict["g2"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 3 (finer)
        self.granularity_alignment_dict["g3"] = nn.Linear(in_features=granularity_dict["g3"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 4 (finer)
        self.granularity_alignment_dict["g4"] = nn.Linear(in_features=granularity_dict["g4"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 5 (finest)
        self.granularity_alignment_dict["g5"] = nn.Linear(in_features=granularity_dict["g5"], out_features=ga_K,
                                                          bias=False).to(device=device)

        # ---- Part 2. Feature Encoding Module (Each Module includes 3 Parts) ---- #
        self.feature_encoder_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward wrong !
        # Granularity 1 (coarsest)
        self.feature_encoder_dict["g1"] = FeatureEncoder(encoding_input_size, encoding_hidden_size).to(device=device)
        # Granularity 2 (fine)
        self.feature_encoder_dict["g2"] = FeatureEncoder(encoding_input_size, encoding_hidden_size).to(device=device)
        # Granularity 3 (finer)
        self.feature_encoder_dict["g3"] = FeatureEncoder(encoding_input_size, encoding_hidden_size).to(device=device)
        # Granularity 4 (finer)
        self.feature_encoder_dict["g4"] = FeatureEncoder(encoding_input_size, encoding_hidden_size).to(device=device)
        # Granularity 5 (finest)
        self.feature_encoder_dict["g5"] = FeatureEncoder(encoding_input_size, encoding_hidden_size).to(device=device)

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

        # ---- Step 0. Get the different granularity feature ---- #
        # - g1 feature (coarsest), shape=(bs, T, K^g1, D)
        feature_g1 = mul_granularity_input["g1"].to(dtype=torch.float32, device=self.device)
        # - g2 feature (fine), shape=(bs, T, K^g2, D)
        feature_g2 = mul_granularity_input["g2"].to(dtype=torch.float32, device=self.device)
        # - g3 feature (finer), shape=(bs, T, K^g3, D)
        feature_g3 = mul_granularity_input["g3"].to(dtype=torch.float32, device=self.device)
        # - g4 feature (finer), shape=(bs, T, K^g4, D)
        feature_g4 = mul_granularity_input["g4"].to(dtype=torch.float32, device=self.device)
        # - g5 feature (finest), shape=(bs, T, K^g5, D)
        feature_g5 = mul_granularity_input["g5"].to(dtype=torch.float32, device=self.device)
        # get the shape
        bs, T = feature_g1.shape[0], feature_g1.shape[1]

        # ---- Step 1. Align the granularity ---- #
        # transpose the feature for alignment
        feature_g1 = feature_g1.permute(0, 1, 3, 2)  # shape from (bs, T, K^g1, D) to (bs, T, D, K^g1)
        feature_g2 = feature_g2.permute(0, 1, 3, 2)  # shape from (bs, T, K^g2, D) to (bs, T, D, K^g2)
        feature_g3 = feature_g3.permute(0, 1, 3, 2)  # shape from (bs, T, K^g3, D) to (bs, T, D, K^g3)
        feature_g4 = feature_g4.permute(0, 1, 3, 2)  # shape from (bs, T, K^g4, D) to (bs, T, D, K^g4)
        feature_g5 = feature_g5.permute(0, 1, 3, 2)  # shape from (bs, T, K^g5, D) to (bs, T, D, K^g5)
        # align the feature with linear transform, get the `F`
        F_g1 = self.granularity_alignment_dict["g1"](feature_g1).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g2 = self.granularity_alignment_dict["g2"](feature_g2).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g3 = self.granularity_alignment_dict["g3"](feature_g3).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g4 = self.granularity_alignment_dict["g4"](feature_g4).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g5 = self.granularity_alignment_dict["g5"](feature_g5).reshape(bs, T, -1)  # shape=(bs, T, D*K)

        # ---- Step 2. Encoding Feature with the residual learning framework ---- #
        # - g1 feature encoding
        P_g1 = F_g1  # the P of granularity is F
        H_g1, y_g1, R_g1 = self.feature_encoder_dict["g1"](P_g1)
        # - g2 feature encoding
        P_g2 = F_g2 - R_g1  # residual learning, shape=(bs, T, D*K)
        H_g2, y_g2, R_g2 = self.feature_encoder_dict["g2"](P_g2)
        # - g3 feature encoding
        P_g3 = F_g3 - R_g2  # residual learning, shape=(bs, T, D*K)
        H_g3, y_g3, R_g3 = self.feature_encoder_dict["g3"](P_g3)
        # - g4 feature encoding
        P_g4 = F_g4 - R_g3  # residual learning, shape=(bs, T, D*K)
        H_g4, y_g4, R_g4 = self.feature_encoder_dict["g4"](P_g4)
        # - g5 feature encoding
        P_g5 = F_g5 - R_g4  # residual learning, shape=(bs, T, D*K)
        H_g5, y_g5, R_g5_ = self.feature_encoder_dict["g5"](P_g5)

        # ---- Step 3. Return ---- #
        # concat the prediction of all granularity
        y_all_g = torch.cat([y_g1, y_g2, y_g3, y_g4, y_g5], dim=1)  # shape=(bs, 5)
        # construct the output and return
        output = {
            "pred": torch.mean(y_all_g, -1, keepdim=True),
            "rec_residuals": (P_g2, P_g3, P_g4, P_g5)
        }
        return output


class MgRL_Attention_Net(nn.Module):
    """ The basic Multi-Granularity Residual Learning Net. """

    def __init__(
            self, granularity_dict: Dict[str, int], ga_K: int,
            encoding_input_size: int, encoding_hidden_size: int, device: torch.device
    ):
        """ The init function of MgRL_Attention Net.

        There are 2 main parts of Multi-Granularity Residual Learning Net:
            - Part 1. Granularity Alignment Module (granularity_alignment): Align the different granularity features to K dim
            - Part 2. Feature Encoding Module (feature_encoder): Encoding the K dim feature and get the 3 outputs

        :param granularity_dict: the dict of input data granularity, should be format like: { "g1": g_1, "g2": g_2, ..., "gG":g_G}
        :param ga_K: the K of Granularity Alignment
        :param encoding_input_size: the input size of encoding feature
        :param encoding_hidden_size: the hidden size of encoding feature
        :param device: the computing device

        """

        super(MgRL_Attention_Net, self).__init__()
        self.device = device

        # ---- Part 1. Granularity Alignment Module (Each Module includes 1 Linear Layer) ---- #
        self.granularity_alignment_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward wrong !
        # Granularity 1 (coarsest)
        self.granularity_alignment_dict["g1"] = nn.Linear(in_features=granularity_dict["g1"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 2 (fine)
        self.granularity_alignment_dict["g2"] = nn.Linear(in_features=granularity_dict["g2"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 3 (finer)
        self.granularity_alignment_dict["g3"] = nn.Linear(in_features=granularity_dict["g3"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 4 (finer)
        self.granularity_alignment_dict["g4"] = nn.Linear(in_features=granularity_dict["g4"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 5 (finest)
        self.granularity_alignment_dict["g5"] = nn.Linear(in_features=granularity_dict["g5"], out_features=ga_K,
                                                          bias=False).to(device=device)

        # ---- Part 2. Feature Encoding Module (Each Module includes 2 Parts) ---- #
        self.feature_encoder_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward wrong !
        # Granularity 1 (coarsest)
        self.feature_encoder_dict["g1"] = NoPred_FeatureEncoder(encoding_input_size, encoding_hidden_size).to(device=device)
        # Granularity 2 (fine)
        self.feature_encoder_dict["g2"] = NoPred_FeatureEncoder(encoding_input_size, encoding_hidden_size).to(device=device)
        # Granularity 3 (finer)
        self.feature_encoder_dict["g3"] = NoPred_FeatureEncoder(encoding_input_size, encoding_hidden_size).to(device=device)
        # Granularity 4 (finer)
        self.feature_encoder_dict["g4"] = NoPred_FeatureEncoder(encoding_input_size, encoding_hidden_size).to(device=device)
        # Granularity 5 (finest)
        self.feature_encoder_dict["g5"] = NoPred_FeatureEncoder(encoding_input_size, encoding_hidden_size).to(device=device)

        # ---- Part 3. Attention Module ---- #
        self.attention = ScaledDotProductAttention().to(device=device)

        # ---- Part 4. Pred Net ---- #
        self.pred_net_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward wrong !
        # Granularity 1 (coarsest)
        self.pred_net_dict["g1"] = nn.Linear(in_features=encoding_hidden_size, out_features=1).to(device=device)
        # Granularity 2 (fine)
        self.pred_net_dict["g2"] = nn.Linear(in_features=encoding_hidden_size, out_features=1).to(device=device)
        # Granularity 3 (finer)
        self.pred_net_dict["g3"] = nn.Linear(in_features=encoding_hidden_size, out_features=1).to(device=device)
        # Granularity 4 (finer)
        self.pred_net_dict["g4"] = nn.Linear(in_features=encoding_hidden_size, out_features=1).to(device=device)
        # Granularity 5 (finest)
        self.pred_net_dict["g5"] = nn.Linear(in_features=encoding_hidden_size, out_features=1).to(device=device)

    def forward(self, mul_granularity_input: Dict[str, torch.Tensor]) -> dict:
        """ The forward function of MgRL_Attention Net.

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

        # ---- Step 0. Get the different granularity feature ---- #
        # - g1 feature (coarsest), shape=(bs, T, K^g1, D)
        feature_g1 = mul_granularity_input["g1"].to(dtype=torch.float32, device=self.device)
        # - g2 feature (fine), shape=(bs, T, K^g2, D)
        feature_g2 = mul_granularity_input["g2"].to(dtype=torch.float32, device=self.device)
        # - g3 feature (finer), shape=(bs, T, K^g3, D)
        feature_g3 = mul_granularity_input["g3"].to(dtype=torch.float32, device=self.device)
        # - g4 feature (finer), shape=(bs, T, K^g4, D)
        feature_g4 = mul_granularity_input["g4"].to(dtype=torch.float32, device=self.device)
        # - g5 feature (finest), shape=(bs, T, K^g5, D)
        feature_g5 = mul_granularity_input["g5"].to(dtype=torch.float32, device=self.device)
        # get the shape
        bs, T = feature_g1.shape[0], feature_g1.shape[1]

        # ---- Step 1. Align the granularity ---- #
        # transpose the feature for alignment
        feature_g1 = feature_g1.permute(0, 1, 3, 2)  # shape from (bs, T, K^g1, D) to (bs, T, D, K^g1)
        feature_g2 = feature_g2.permute(0, 1, 3, 2)  # shape from (bs, T, K^g2, D) to (bs, T, D, K^g2)
        feature_g3 = feature_g3.permute(0, 1, 3, 2)  # shape from (bs, T, K^g3, D) to (bs, T, D, K^g3)
        feature_g4 = feature_g4.permute(0, 1, 3, 2)  # shape from (bs, T, K^g4, D) to (bs, T, D, K^g4)
        feature_g5 = feature_g5.permute(0, 1, 3, 2)  # shape from (bs, T, K^g5, D) to (bs, T, D, K^g5)
        # align the feature with linear transform, get the `F`
        F_g1 = self.granularity_alignment_dict["g1"](feature_g1).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g2 = self.granularity_alignment_dict["g2"](feature_g2).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g3 = self.granularity_alignment_dict["g3"](feature_g3).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g4 = self.granularity_alignment_dict["g4"](feature_g4).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g5 = self.granularity_alignment_dict["g5"](feature_g5).reshape(bs, T, -1)  # shape=(bs, T, D*K)

        # ---- Step 2. Encoding Feature with the residual learning framework ---- #
        # - g1 feature encoding
        P_g1 = F_g1  # the P of granularity is F
        H_g1, R_g1 = self.feature_encoder_dict["g1"](P_g1)
        # - g2 feature encoding
        P_g2 = F_g2 - R_g1  # residual learning, shape=(bs, T, D*K)
        H_g2, R_g2 = self.feature_encoder_dict["g2"](P_g2)
        # - g3 feature encoding
        P_g3 = F_g3 - R_g2  # residual learning, shape=(bs, T, D*K)
        H_g3, R_g3 = self.feature_encoder_dict["g3"](P_g3)
        # - g4 feature encoding
        P_g4 = F_g4 - R_g3  # residual learning, shape=(bs, T, D*K)
        H_g4, R_g4 = self.feature_encoder_dict["g4"](P_g4)
        # - g5 feature encoding
        P_g5 = F_g5 - R_g4  # residual learning, shape=(bs, T, D*K)
        H_g5, R_g5_ = self.feature_encoder_dict["g5"](P_g5)

        # ---- Step 3. Soft attention of last step hidden ---- #
        # be careful about the concat sequence, shape=(bs, 5, hidden_size)
        H_last_step_g = torch.cat([H_g1[:, -1:, :], H_g2[:, -1:, :], H_g3[:, -1:, :], H_g4[:, -1:, :], H_g5[:, -1:, :]], dim=1)
        # do the attention
        H_attention_g = self.attention(queries=H_last_step_g, keys=H_last_step_g, values=H_last_step_g)

        # ---- Step 4. Get the prediction ---- #
        y_g1 = self.pred_net_dict["g1"](H_attention_g[:, 0, :])  # for g1
        y_g2 = self.pred_net_dict["g2"](H_attention_g[:, 1, :])  # for g2
        y_g3 = self.pred_net_dict["g3"](H_attention_g[:, 2, :])  # for g3
        y_g4 = self.pred_net_dict["g4"](H_attention_g[:, 3, :])  # for g4
        y_g5 = self.pred_net_dict["g5"](H_attention_g[:, 4, :])  # for g5

        # ---- Step 5. Return ---- #
        # concat the prediction of all granularity
        y_all_g = torch.cat([y_g1, y_g2, y_g3, y_g4, y_g5], dim=1)  # shape=(bs, 5)
        # construct the output and return
        output = {
            "pred": torch.mean(y_all_g, -1, keepdim=True),
            "rec_residuals": (P_g2, P_g3, P_g4, P_g5)
        }
        return output


class MgRL_CE_Net(nn.Module):
    """ The Multi-granularity Residual Learning Framework with Confidence Estimation (CE). """

    def __init__(
            self, granularity_dict: Dict[str, int], ga_K: int,
            encoding_input_size: int, encoding_hidden_size: int, negative_sample_num: int, device: torch.device
    ):
        """ The init function of MgRL_CE_Net.

        There are 2 main parts of Multi-Granularity Residual Learning Net:
            - Part 1. Granularity Alignment Module (granularity_alignment): Align the different granularity features to K dim
            - Part 2. Feature Encoding With CE Module (feature_encoder): Encoding the K dim feature and get the 3 outputs

        :param granularity_dict: the dict of input data granularity, should be format like: { "g1": g_1, "g2": g_2, ..., "gG":g_G}
        :param ga_K: the K of Granularity Alignment
        :param encoding_input_size: the input size of encoding feature
        :param encoding_hidden_size: the hidden size of encoding feature
        :param negative_sample_num: the number of negative samples
        :param device: the computing device

        """

        super(MgRL_CE_Net, self).__init__()
        self.device = device

        # ---- Part 1. Granularity Alignment Module (Each Module includes 1 Linear Layer) ---- #
        self.granularity_alignment_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward wrong !
        # Granularity 1 (coarsest)
        self.granularity_alignment_dict["g1"] = nn.Linear(in_features=granularity_dict["g1"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 2 (fine)
        self.granularity_alignment_dict["g2"] = nn.Linear(in_features=granularity_dict["g2"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 3 (finer)
        self.granularity_alignment_dict["g3"] = nn.Linear(in_features=granularity_dict["g3"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 4 (finer)
        self.granularity_alignment_dict["g4"] = nn.Linear(in_features=granularity_dict["g4"], out_features=ga_K,
                                                          bias=False).to(device=device)
        # Granularity 5 (finest)
        self.granularity_alignment_dict["g5"] = nn.Linear(in_features=granularity_dict["g5"], out_features=ga_K,
                                                          bias=False).to(device=device)

        # ---- Part 2. Feature Encoding Module (Each Module includes 3 Parts) ---- #
        self.feature_encoder_ce_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward wrong !
        # Granularity 1 (coarsest)
        self.feature_encoder_ce_dict["g1"] = FeatureEncoderCE(encoding_input_size, encoding_hidden_size,
                                                              negative_sample_num).to(device=device)
        # Granularity 2 (fine)
        self.feature_encoder_ce_dict["g2"] = FeatureEncoderCE(encoding_input_size, encoding_hidden_size,
                                                              negative_sample_num).to(device=device)
        # Granularity 3 (finer)
        self.feature_encoder_ce_dict["g3"] = FeatureEncoderCE(encoding_input_size, encoding_hidden_size,
                                                              negative_sample_num).to(device=device)
        # Granularity 4 (finer)
        self.feature_encoder_ce_dict["g4"] = FeatureEncoderCE(encoding_input_size, encoding_hidden_size,
                                                              negative_sample_num).to(device=device)
        # Granularity 5 (finest)
        self.feature_encoder_ce_dict["g5"] = FeatureEncoderCE(encoding_input_size, encoding_hidden_size,
                                                              negative_sample_num).to(device=device)

    def forward(self, mul_granularity_input: Dict[str, torch.Tensor]) -> dict:
        """ The forward function of MgRL_CE_Net.

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
                "rec_residuals" : a tuple of reconstruction residual, each residual have the same shape=(bs, T, D*K),
                "contrastive_loss" : a tensor of the trend contrastive loss, shape=(bs, 1)
            }

        """

        # ---- Step 0. Get the different granularity feature ---- #
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
        # get the shape
        bs, T = feature_g1.shape[0], feature_g1.shape[1]

        # ---- Step 1. Align the granularity ---- #
        # transpose the feature for alignment
        feature_g1 = feature_g1.permute(0, 1, 3, 2)  # shape from (bs, T, K^g1, D) to (bs, T, D, K^g1)
        feature_g2 = feature_g2.permute(0, 1, 3, 2)  # shape from (bs, T, K^g2, D) to (bs, T, D, K^g2)
        feature_g3 = feature_g3.permute(0, 1, 3, 2)  # shape from (bs, T, K^g3, D) to (bs, T, D, K^g3)
        feature_g4 = feature_g4.permute(0, 1, 3, 2)  # shape from (bs, T, K^g4, D) to (bs, T, D, K^g4)
        feature_g5 = feature_g5.permute(0, 1, 3, 2)  # shape from (bs, T, K^g5, D) to (bs, T, D, K^g5)
        # align the feature with linear transform, get the `F`
        F_g1 = self.granularity_alignment_dict["g1"](feature_g1).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g2 = self.granularity_alignment_dict["g2"](feature_g2).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g3 = self.granularity_alignment_dict["g3"](feature_g3).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g4 = self.granularity_alignment_dict["g4"](feature_g4).reshape(bs, T, -1)  # shape=(bs, T, D*K)
        F_g5 = self.granularity_alignment_dict["g5"](feature_g5).reshape(bs, T, -1)  # shape=(bs, T, D*K)

        # ---- Step 2. Encoding Feature with the residual learning framework ---- #
        # - g1 feature encoding
        P_g1 = F_g1  # the P of granularity is F
        H_g1, y_g1, R_g1, alpha_g1, con_l_g1 = self.feature_encoder_ce_dict["g1"](P_g1)
        # - g2 feature encoding
        P_g2 = F_g2 - R_g1  # residual learning, shape=(bs, T, D*K)
        H_g2, y_g2, R_g2, alpha_g2, con_l_g2 = self.feature_encoder_ce_dict["g2"](P_g2)
        # - g3 feature encoding
        P_g3 = F_g3 - R_g2  # residual learning, shape=(bs, T, D*K)
        H_g3, y_g3, R_g3, alpha_g3, con_l_g3 = self.feature_encoder_ce_dict["g3"](P_g3)
        # - g4 feature encoding
        P_g4 = F_g4 - R_g3  # residual learning, shape=(bs, T, D*K)
        H_g4, y_g4, R_g4, alpha_g4, con_l_g4 = self.feature_encoder_ce_dict["g4"](P_g4)
        # - g5 feature encoding
        P_g5 = F_g5 - R_g4  # residual learning, shape=(bs, T, D*K)
        H_g5, y_g5, R_g5, alpha_g5, con_l_g5 = self.feature_encoder_ce_dict["g5"](P_g5)

        # ---- Step 3. Return ---- #
        # concat the prediction of all granularity
        y_all_g = torch.cat([y_g1, y_g2, y_g3, y_g4, y_g5], dim=1)  # shape=(bs, 5)
        # concat the alpha of all granularity
        alpha_all_g = torch.cat([alpha_g1, alpha_g2, alpha_g3, alpha_g4, alpha_g5], dim=1)  # shape=(bs, 5)
        alpha_all_g = F.softmax(alpha_all_g, dim=1)  # use the soft_max to weight, shape=(bs, 5)
        # use the alpha to weight the y
        y_all_g = y_all_g * alpha_all_g
        # construct the output and return, contrastive_loss is sum of all granularity
        output = {
            "pred": torch.mean(y_all_g, -1, keepdim=True),
            "rec_residuals": (P_g2, P_g3, P_g4, P_g5),
            "contrastive_loss": con_l_g1 + con_l_g2 + con_l_g3 + con_l_g4 + con_l_g5
        }
        return output


if __name__ == "__main__":
    bath_size, time_steps, D = 64, 4, 1
    mg_input = {
        "g1": torch.ones((bath_size, time_steps, 1, D)),
        "g2": torch.ones((bath_size, time_steps, 2, D)),
        "g3": torch.ones((bath_size, time_steps, 6, D)),
        "g4": torch.ones((bath_size, time_steps, 24, D)),
        "g5": torch.ones((bath_size, time_steps, 96, D))
    }
    g_dict = {"g1": 1, "g2": 2, "g3": 6, "g4": 24, "g5": 96}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MgRL_CE_Net(granularity_dict=g_dict, ga_K=2, encoding_input_size=2, encoding_hidden_size=7, device=dev, negative_sample_num=5)
    out = model(mg_input)
    print(out["pred"].shape)
    # print(out["rec_residuals"])
