# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 10:37
# @Author  : Karry Ren

""" The loss functions:
        - MSE_Loss
        - MgRL_Loss
        - MgRL_CE_Loss
"""

import torch


class MSE_Loss:
    """ Compute the MSE loss.

    loss = reduction((y_true - y_pred)^2)

    """

    def __init__(self, reduction: str = "mean"):
        """ Init function of the MSE Loss.

        :param reduction: the reduction way of this loss, you have only 2 choices now:
            - `sum` for sum reduction
            - `mean` for mean reduction

        """

        assert reduction in ["sum", "mean"], f"Reduction in MgRL_Loss ERROR !! `{reduction}` is not allowed !!"
        self.reduction = reduction  # the reduction way

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, weight: torch.Tensor):
        """ Call function of the MSE Loss.

        :param y_true: the true label of time series prediction, shape=(bs, 1)
        :param y_pred: the prediction, shape=(bs, 1)
        :param weight: the weight indicates item meaningful or meaningless, shape=(bs, 1)

        return:
            - batch_loss: a Tensor number, shape=([])

        """

        # ---- Step 0. Test the weight shape & make the default weight ---- #
        assert weight.shape[0] == y_true.shape[0], "Weight should have the same length with y_true&y_pred !"

        # ---- Step 1. Compute the loss ---- #
        if self.reduction == "mean":
            # compute mse loss (`mean`)
            mse_sample_loss = (y_pred - y_true) ** 2  # shape=(bs, 1)
            mse_loss = torch.sum(weight * mse_sample_loss) / torch.sum(weight)  # weighted and mean
            batch_loss = mse_loss
        elif self.reduction == "sum":
            # compute mse loss (`sum`)
            mse_sample_loss = (y_pred - y_true) ** 2  # shape=(bs, 1)
            mse_loss = torch.sum((weight * mse_sample_loss))  # weighted and sum
            batch_loss = mse_loss
        else:
            raise TypeError(self.reduction)

        # ---- Step 2. Return loss ---- #
        return batch_loss


class DeepAR_Loss:
    """ Compute using gaussian the log-likelihood which needs to be maximized.

    TODO: only likelihood is not work !

    """

    def __init__(self, reduction: str = "mean"):
        """ Init function of the MSE Loss.

        :param reduction: the reduction way of this loss, you have only 2 choices now:
            - `sum` for sum reduction
            - `mean` for mean reduction

        """

        assert reduction in ["sum", "mean"], f"Reduction in MgRL_Loss ERROR !! `{reduction}` is not allowed !!"
        self.reduction = reduction  # the reduction way
        self.mse = MSE_Loss(reduction=reduction)  # get the mse loss

    def __call__(self, y_true: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, weight: torch.Tensor):
        """ Call function of the DeepAR Loss.

        :param y_true: the true label of time series prediction, shape=(bs, 1)
        :param mu: the prediction mu, shape=(bs, 1)
        :param sigma: the prediction sigma, shape=(bs, 1)
        :param weight: the weight indicates item meaningful or meaningless, shape=(bs, 1)

        return:
            - batch_loss: a Tensor number, shape=([])

        """

        # ---- Step 0. Test the weight shape & make the default weight ---- #
        assert weight.shape[0] == y_true.shape[0], "Weight should have the same length with y_true&y_pred !"

        # ---- Step 1. Get the meaningful data index ---- #
        meaningful_index = (weight != 0.0)

        # ---- Step 2. Compute the distribution loss ---- #
        distribution = torch.distributions.normal.Normal(mu[meaningful_index], sigma[meaningful_index])
        likelihood = distribution.log_prob(y_true[meaningful_index])
        if self.reduction == "mean":
            # compute likelihood mean
            likelihood_loss = -torch.mean(likelihood)
        elif self.reduction == "sum":
            # compute likelihood std
            likelihood_loss = -torch.sum(likelihood)
        else:
            raise TypeError(self.reduction)

        # ---- Step 3. Add the mse loss ---- #
        batch_loss = likelihood_loss + self.mse(y_true=y_true, y_pred=mu, weight=weight)

        # ---- Step 4. Return loss ---- #
        return batch_loss


class MgRL_Loss:
    """ Compute the loss of MgRLNet, which has two parts:
        - part 1. MSE Loss
        - part 2. Reconstruction Loss

    The detail computing function please see my paper.

    """

    def __init__(self, reduction: str = "sum", lambda_1: float = 1.0):
        """ Init function of the MgRL Loss.

        :param reduction: the reduction way of this loss, you have only 2 choices now:
            - `sum` for sum reduction
            - `mean` for mean reduction
        :param lambda_1: the hyper-param lambda_1 of MgRL Loss

        NOTE: You might have question about the weight decay and where is the lambda_theta of the loss.
            All settings about the weight_decay are in optimizer !

        """

        assert reduction in ["sum", "mean"], f"Reduction in MgRL_Loss ERROR !! `{reduction}` is not allowed !!"
        self.reduction = reduction  # the reduction way
        self.lambda_1 = lambda_1  # the lambda weight of Reconstruction Loss

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, rec_residuals: tuple, weight: torch.Tensor):
        """ Call function of the MgRL Loss.

        :param y_true: the true label of time series prediction, shape=(bs, 1)
        :param y_pred: the prediction of MgRLNet, shape=(bs, 1)
        :param rec_residuals: the rec_residuals of MgRLNet, a tuple of (bs, T, D*K)
        :param weight: the weight indicates item meaningful or meaningless, shape=(bs, 1)

        return:
            - batch_loss: a Tensor number, shape=([])

        """

        # ---- Step 0. Test the weight shape & make the default weight ---- #
        assert weight.shape[0] == y_true.shape[0], "Weight should have the same length with y_true&y_pred !"
        bs, device = y_pred.shape[0], y_pred.device

        # ---- Step 1. Compute the loss ---- #
        if self.reduction == "mean":
            # compute mse loss (`mean`)
            mse_sample_loss = (y_pred - y_true) ** 2  # shape=(bs, 1)
            mse_loss = torch.sum(weight * mse_sample_loss) / torch.sum(weight)  # weighted and mean
            # compute rec loss (`mean`)
            rec_sample_loss = torch.zeros((bs, 1)).to(dtype=torch.float32, device=device)  # shape=(bs, 1)
            for rec_res in rec_residuals:  # for loop to compute rec loss of each sample
                rec_sample_loss += torch.sum(rec_res ** 2, dim=(1, 2), keepdim=True).reshape(bs, 1)
            rec_loss = torch.sum(weight * rec_sample_loss) / torch.sum(weight)  # weighted and mean
            # sum the mse loss and rec loss
            batch_loss = mse_loss + self.lambda_1 * rec_loss
        elif self.reduction == "sum":
            # compute mse loss (`sum`)
            mse_sample_loss = (y_pred - y_true) ** 2  # shape=(bs, 1)
            mse_loss = torch.sum((weight * mse_sample_loss))  # weighted and sum
            # compute rec loss (`sum`)
            rec_sample_loss = torch.zeros((bs, 1)).to(dtype=torch.float32, device=device)  # shape=(bs, 1)
            for rec_res in rec_residuals:  # for loop to compute rec loss of each sample
                rec_sample_loss += torch.sum(rec_res ** 2, dim=(1, 2), keepdim=True).reshape(bs, 1)
            rec_loss = torch.sum(weight * rec_sample_loss)  # weighted and sum
            # sum the mse loss and rec loss
            batch_loss = mse_loss + self.lambda_1 * rec_loss
        else:
            raise TypeError(self.reduction)

        # ---- Step 2. Return loss ---- #
        return batch_loss


class MgRL_CE_Loss:
    """ Compute the loss of MgRL_CE_Net, which has two parts:
        - part 1. MSE Loss
        - part 2. Reconstruction Loss
        - part 3. Contrastive Loss

    The detail computing function please see my paper.

    """

    def __init__(self, reduction: str = "sum", lambda_1: float = 1.0, lambda_2: float = 1.0):
        """ Init function of the MgRL Loss.

        :param reduction: the reduction way of this loss, you have only 2 choices now:
            - `sum` for sum reduction
            - `mean` for mean reduction
        :param lambda_1: the hyper-param lambda_1 of MgRL_CE Loss
        :param lambda_2: the hyper-param lambda_2 of MgRL_CE Loss

        NOTE: You might have question about the weight decay and where is the lambda_theta of the loss.
            All settings about the weight_decay are in optimizer !

        """

        assert reduction in ["sum", "mean"], f"Reduction in MgRL_Loss ERROR !! `{reduction}` is not allowed !!"
        self.reduction = reduction  # the reduction way
        self.lambda_1 = lambda_1  # the lambda weight of Reconstruction Loss
        self.lambda_2 = lambda_2  # the lambda weight of Contrastive Loss

    def __call__(
            self, y_true: torch.Tensor, y_pred: torch.Tensor, rec_residuals: tuple,
            contrastive_loss: torch.Tensor, weight: torch.Tensor
    ):
        """ Call function of the MgRL_CE Loss.

        :param y_true: the true label of time series prediction, shape=(bs, 1)
        :param y_pred: the prediction of MgRL_CE_Net, shape=(bs, 1)
        :param rec_residuals: the rec_residuals of MgRL_CE_Net, a tuple of (bs, T, D*K)
        :param contrastive_loss: the contrastive_loss of the MgRL_CE_Net, shape=(bs, 1)
        :param weight: the weight indicates item meaningful or meaningless, shape=(bs, 1)

        return:
            - batch_loss: a Tensor number, shape=([])

        """

        # ---- Step 0. Test the weight shape & make the default weight ---- #
        assert weight.shape[0] == y_true.shape[0], "Weight should have the same length with y_true&y_pred !"
        bs, device = y_pred.shape[0], y_pred.device

        # ---- Step 1. Compute the loss ---- #
        if self.reduction == "mean":
            # compute mse loss (`mean`)
            mse_sample_loss = (y_pred - y_true) ** 2  # shape=(bs, 1)
            mse_loss = torch.sum(weight * mse_sample_loss) / torch.sum(weight)  # weighted and mean
            # compute rec loss (`mean`)
            rec_sample_loss = torch.zeros((bs, 1)).to(dtype=torch.float32, device=device)  # shape=(bs, 1)
            for rec_res in rec_residuals:  # for loop to compute rec loss of each sample
                rec_sample_loss += torch.sum(rec_res ** 2, dim=(1, 2), keepdim=True).reshape(bs, 1)
            rec_loss = torch.sum(weight * rec_sample_loss) / torch.sum(weight)  # weighted and mean
            # compute contrastive loss (`mean`)
            contrastive_sample_loss = contrastive_loss  # shape=(bs, 1)
            contra_loss = torch.sum(weight * contrastive_sample_loss) / torch.sum(weight)  # weighted and mean
        elif self.reduction == "sum":
            # compute mse loss (`sum`)
            mse_sample_loss = (y_pred - y_true) ** 2  # shape=(bs, 1)
            mse_loss = torch.sum((weight * mse_sample_loss))  # weighted and sum
            # compute rec loss (`sum`)
            rec_sample_loss = torch.zeros((bs, 1)).to(dtype=torch.float32, device=device)  # shape=(bs, 1)
            for rec_res in rec_residuals:  # for loop to compute rec loss of each sample
                rec_sample_loss += torch.sum(rec_res ** 2, dim=(1, 2), keepdim=True).reshape(bs, 1)
            rec_loss = torch.sum(weight * rec_sample_loss)  # weighted and sum
            # compute contrastive loss (`sum`)
            contrastive_sample_loss = contrastive_loss  # shape=(bs, 1)
            contra_loss = torch.sum(weight * contrastive_sample_loss)  # weighted and sum
        else:
            raise TypeError(self.reduction)

        # ---- Step 2. Return loss ---- #
        # sum the mse loss and rec loss
        print(mse_loss, rec_loss, contra_loss)
        batch_loss = mse_loss + self.lambda_1 * rec_loss + self.lambda_2 * contra_loss
        return batch_loss


if __name__ == "__main__":
    # An Example test two loss
    bs, T, D = 64, 3, 2
    y_true, y_pred, weight = torch.zeros((bs, 1)), torch.ones((bs, 1)), torch.ones((bs, 1))
    a = torch.ones((bs, T, D))
    rec_residuals = (torch.zeros((bs, T, D)), a, torch.zeros((bs, T, D)), torch.zeros((bs, T, D)))

    # ---- Test MSE_Loss ---- #
    loss_mse_sum = MSE_Loss(reduction="sum")
    l_sum = loss_mse_sum(y_true=y_true, y_pred=y_pred, weight=weight)
    print(l_sum)
    assert l_sum == bs * 1
    loss_mse_mean = MSE_Loss(reduction="mean")
    l_sum = loss_mse_mean(y_true=y_true, y_pred=y_pred, weight=weight)
    print(l_sum)
    assert l_sum == bs / weight.sum()
