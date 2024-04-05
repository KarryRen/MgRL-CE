# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 10:37
# @Author  : Karry Ren

""" The loss function of MgRLNet and MgRL_CE_Net"""

import torch


class MgRL_Loss:
    """ Compute the loss of MgRL, which has two parts:
        - part 1. MSE Loss
        - part 2. Reconstruction Loss

    """

    def __init__(self, reduction: str = "sum", lambda_1: float = 1.0):
        assert reduction in ["sum", "mean"], f"Reduction in MgRL_Loss ERROR !! `{reduction}` is not allowed !!"
        self.reduction = reduction  # the reduction
        self.lambda_1 = lambda_1  # the lambda weight of Reconstruction Loss

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, rec_residuals: tuple, weight: torch.Tensor):
        """ Compute the MgRL Loss.

        :param y_true: the true label of time series prediction, shape=(bs, 1)
        :param y_pred: the prediction of MgRL Net, shape=(bs, 1)
        :param

        """

        # ---- Step 1. Test the weight shape & make the default weight ---- #
        assert weight.shape[0] == y_true.shape[0], "Weight should have the same length with y_true&y_pred !"

        # ---- Step 2. Compute the batch loss ---- #
        if self.reduction == "mean":
            batch_loss = torch.sum(weight * torch.sum((y_pred - y_true) ** 2, dim=1, keepdim=True)) / torch.sum(weight)
        elif self.reduction == "sum":
            batch_loss = torch.sum((weight * (y_pred - y_true)) ** 2)
