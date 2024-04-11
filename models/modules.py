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
import torch.nn.functional as F


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

    def __init__(self, input_size: int, hidden_size: int, negative_sample_num: int):
        """ The init function of FeatureEncoder.

        :param input_size: the input size of encoding feature
        :param hidden_size: the hidden size of encoding feature
        :param negative_sample_num: the num of negative sample num (`N -1` in my paper)

        """

        super(FeatureEncoderCE, self).__init__()
        self.neg_sample_num = negative_sample_num

        # ---- Part 1. F_{TEnc}: Temporal Feature Encoder ---- #
        self.temporal_feature_encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)

        # ---- Part 2. W: The weight of Discriminator ---- #
        self.W = nn.Linear(in_features=hidden_size, out_features=input_size, bias=False)

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

    def forward(self, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """ The forward function of FeatureEncoder.

        :param P: the P of each granularity, shape=(bs, T, D*K), where D*K is the encoding_input_size
            P_1 = F_1 && P_other(g) = F_other(g) - R_(g-1)

        returns:
            - H: the hidden state of each granularity, shape=(bs, T, hidden_size)
            - y: the prediction of each granularity, shape=(bs, 1)
            - R: the reconstruction of each granularity, shape=(bs, T, D*K|input_size)
            - alpha: the alpha of each granularity, shape=(bs, 1)
            - trend_contrastive_loss: the contrastive loss of each granularity, shape=()

        """

        # ---- Step 0. Get the `T` ---- #
        T = P.shape[1]

        # ---- Step 1. Use the temporal_feature_encoder to encode P ---- #
        H, _ = self.temporal_feature_encoder(P)  # shape=(bs, T, hidden_size)

        # ---- Step 2. For loop each step to compute the contrastive loss ---- #
        # init the trend_contrastive_loss
        trend_contrastive_loss = 0.0
        # for loop to compute the loss
        for t in range(1, T):
            # slice the P in time_step t, p_t which represents the `current status` in my paper
            p_t = P[:, t:t + 1, :]  # shape=(bs, 1, hidden_size)
            # slice the H in time_step t-1, C_t which represents the `history trend` in my paper
            C_t = H[:, t - 1:t, :]  # shape=(bs, 1, hidden_size)
            # random select the negative sample, the FIRST 1 is positive.
            pos_neg_p_t = self._random_select_neg_sample(positive_p=p_t)  # (bs, 1+neg_sample_num, hidden_size)
            # get the discriminator score, which is different from paper
            D_p_c = torch.mean(self.W(C_t) * pos_neg_p_t, -1)  # broadcast, shape=(bs,  1+neg_sample_num)
            # compute the trend_contrastive_loss
            pos_ce_loss = F.log_softmax(D_p_c, dim=1)[:, 0]  # only the get the cross entropy loss of the positive sample
            trend_contrastive_loss += -torch.mean(pos_ce_loss)  # use `mean` to do the reduction
        # use the last step (p_T, C_T) to compute alpha
        p_last_step = P[:, -1:, :]  # last step P, shape=(bs, 1, hidden_size)
        C_last_step = H[:, -2:-1, :]  # last step C, shape=(bs, 1, hidden_size)
        alpha = torch.mean(self.W(C_last_step) * p_last_step, -1)  # the al

        # ---- Step 3. Use the prediction_net to get the prediction ---- #
        y = self.prediction_net(H[:, -1, :])  # shape=(bs, 1)

        # ---- Step 4. Use the reconstruction_net to get the reconstruction ---- #
        R = self.reconstruction_net(H)  # shape=(bs, T, D*K)

        # ---- Step 4. Return ---- #
        return H, y, R, alpha, trend_contrastive_loss

    def _random_select_neg_sample(self, positive_p: torch.Tensor):
        """ Generate the negative data from the same mini-batch.

        :param positive_p: the positive p (in P) of one time step, shape=(bs, 1, hidden_size)

        return:
            - pos_neg_p, the positive and negative p of one time step,
                shape=(bs, 1+neg_sample_num, hidden_size) and the FIRST 1 is positive.

        """

        # ---- Get the bs ---- #
        bs = positive_p.shape[0]

        # ---- Define the positive and negative p of one time step ---- #
        pos_neg_p = positive_p.clone()  # init as the positive_p, shape=(bs, 1, hidden_size)

        # ---- For loop neg_sample_num times to generate neg_sample_num ---- #
        for _ in range(self.neg_sample_num):
            # shuffle the batch idx [0, bs)
            negative_batch_idx_list = torch.randperm(bs)
            # the negative_batch_idx_list MUST be ensured that every element is not
            # equal to the original batch_idx, which means for every `bi`
            # the negative_batch_idx_list[bi] != bi. But only using `randperm()` is not enough !!!
            for bi in range(len(negative_batch_idx_list)):
                if negative_batch_idx_list[bi] != bi:  # greate, negative_batch_idx_list[bi] != bi
                    continue
                elif bi == 0:  # negative_batch_idx_list[bi] == bi and bi == 0, then chose from [1, bs)
                    negative_batch_idx_list[bi] = torch.randint(1, bs, (1,))[0]
                else:  # negative_batch_idx_list[bi] == bi and bi != 0, then chose from [0, bi)
                    negative_batch_idx_list[bi] = torch.randint(0, bi, (1,))[0]
            # check the negative_batch_idx_list[bi] != bi
            assert (negative_batch_idx_list - torch.arange(bs) != 0).all(), "negative_batch_idx_list ERROR !!!"
            # select the negative samples
            negative_p = positive_p[negative_batch_idx_list]
            pos_neg_p = torch.cat((pos_neg_p, negative_p), 1)

        # ---- Return the positive and negative p ---- #
        return pos_neg_p


if __name__ == "__main__":
    bs, T, input_size, hidden_size = 2, 5, 4, 8
    P1 = torch.ones((2, 5, 4))
    P1[1] = 0
    P1[1, 1] = 2
    P1[0, 1] = 3

    # ---- Test the feature encoder ---- #
    # feature_encoder = FeatureEncoder(input_size=input_size, hidden_size=hidden_size)
    # H, y, R = feature_encoder(P1)

    # ---- Test the feature encoder with Confidence Estimating (CE) ---- #
    feature_encoder_ce = FeatureEncoderCE(input_size=input_size, hidden_size=hidden_size, negative_sample_num=3)
    H, y, R, alpha, c_loss = feature_encoder_ce(P1)
    print(alpha * y)
