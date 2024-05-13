# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 13:48
# @Author  : Karry Ren

""" The metrics of y_ture and y_pred.
        - corr_score: the Pearson correlation coefficients.
        - rmse_score: the Root Mean Square Error (RMSE).
        - mae_score: the Mean Absolute Error (MAE).

"""

import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray = None, epsilon: float = 1e-10):
    """ :math:`R^2` (coefficient of determination) regression score function.
    :math:`R^2 = 1 - SSR/SST`.

    Best possible score is 1.0, and it can be NEGATIVE (because the model can be arbitrarily worse,
    it need not actually be the square of a quantity R).

    :param y_true: the label, shape=(num_of_samples)
    :param y_pred: the prediction, shape=(num_of_samples)
    :param weight: the weight of label, corr with sample, shape=(num_of_samples) CAN'T BE ALL ZERO !!
    :param epsilon: the epsilon to avoid 0 denominator

     return:
        - r2, a number shape=()

    """

    # ---- Step 1. Test the shape and weight ---- #
    assert y_true.shape == y_pred.shape, f"`y_true`, `y_pred` should have the SAME shape !"
    assert weight.shape == y_true.shape, f"`weight` should have the SAME shape as y_true&y_pred !"
    assert np.sum(weight) > 0, f"weight can't be all zero !"

    # ---- Step 2. Compute the SSR & SSE ---- #
    # compute: SSR = sum(weight * (y - y_hat)^2)), a number, shape=()
    ssr = np.sum(weight * ((y_true - y_pred) ** 2), axis=0, dtype=np.float32)
    # compute SST = sum((y - y_bar)^2)
    # - the weighted mean of y_true, shape=(num_of_samples)
    y_bar = np.sum(weight * y_true, axis=0, keepdims=True) / np.sum(weight, axis=0, keepdims=True)
    # - compute
    sst = np.sum(weight * ((y_true - y_bar) ** 2), axis=0, dtype=np.float32)

    # ---- Step 3. Compute and Return r2 = 1 - SSR/SST ---- #
    r2 = 1 - (ssr / (sst + epsilon))
    return r2


def corr_score(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray, epsilon: float = 1e-10):
    """ The Pearson correlation coefficients.

    :math:`CORR = E[(y_true - y_true_bar)(y_pred - y_pred_bar)] / (std(y_true)*std(y_pred))`
    here we multiply `n - 1` in both numerator and denominator to get:
        corr = sum((y_true - y_true_bar)(y_pred - y_pred_bar)) /
                [sqrt(sum((y_true - y_true_bar) ** 2)) * sqrt(sum((y_pred - y_pred_bar) ** 2))]

    The corr could be [-1.0, 1.0]:
        - the `0` means NO corr
        - `1` means STRONG POSITIVE corr
        - `-1` means STRONG NEGATIVE corr.

    :param y_true: the label, shape=(num_of_samples)
    :param y_pred: the prediction, shape=(num_of_samples)
    :param weight: the weight of label, corr with sample, shape=(num_of_samples) CAN'T BE ALL ZERO !!
    :param epsilon: the epsilon to avoid 0 denominator

    return:
        - corr, a number shape=()

    """

    # ---- Step 1. Test the shape and weight ---- #
    assert y_true.shape == y_pred.shape, f"`y_true`, `y_pred` should have the SAME shape !"
    assert weight.shape == y_true.shape, f"`weight` should have the SAME shape as y_true&y_pred !"
    assert np.sum(weight) > 0, f"weight can't be all zero !"

    # ---- Step 2. Compute numerator & denominator of CORR ---- #
    # compute the weighted mean of y_ture and y_pred, shape=(num_of_samples)
    y_true_bar = np.sum(weight * y_true, axis=0, keepdims=True) / np.sum(weight, axis=0, keepdims=True)
    y_pred_bar = np.sum(weight * y_pred, axis=0, keepdims=True) / np.sum(weight, axis=0, keepdims=True)
    # compute numerator, shape=(), a number
    numerator = np.sum(weight * ((y_true - y_true_bar) * (y_pred - y_pred_bar)), axis=0, dtype=np.float32)
    # compute denominator, shape=(), a number
    sum_y_true_std = np.sqrt(np.sum(weight * ((y_true - y_true_bar) ** 2), axis=0, dtype=np.float32))
    sum_y_pred_std = np.sqrt(np.sum(weight * ((y_pred - y_pred_bar) ** 2), axis=0, dtype=np.float32))
    denominator = sum_y_true_std * sum_y_pred_std

    # ---- Step 3. Compute and Return CORR score ---- #
    corr = numerator / (denominator + epsilon)
    return corr


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray):
    """ The Root Mean Square Error (RMSE).

    :math:`rmse = sqrt(mean((y_true-y_pred)^2))`

    The corr could be [0, +\inf]:
        - the `0` means y_true == y_pred, absolutely !
        - the bigger rmse means the bigger error.

    :param y_true: the label, shape=(num_of_samples)
    :param y_pred: the prediction, shape=(num_of_samples)
    :param weight: the weight of label, corr with sample, shape=(num_of_samples) CAN'T BE ALL ZERO !!

    return:
        - rmse, a number shape=()

    """

    # ---- Step 1. Test the shape and weight ---- #
    assert y_true.shape == y_pred.shape, f"`y_true`, `y_pred` should have the SAME shape !"
    assert weight.shape == y_true.shape, f"`weight` should have the SAME shape as y_true&y_pred !"
    assert np.sum(weight) > 0, f"weight can't be all zero !"

    # ---- Step 2. Compute RMSE ---- #
    rmse = np.sqrt(np.sum(weight * (y_true - y_pred) ** 2) / np.sum(weight))  # weighted and mean

    # ---- Step 3. Return RMSE score ---- #
    return rmse


def mae_score(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray):
    """ The Mean Absolute Error (MAE).

    :math:`mae = mean(|y_true-y_pred|)`

    The corr could be [0, +\inf]:
        - the `0` means y_true == y_pred, absolutely !
        - the bigger mae means the bigger error.

    :param y_true: the label, shape=(num_of_samples)
    :param y_pred: the prediction, shape=(num_of_samples)
    :param weight: the weight of label, corr with sample, shape=(num_of_samples) CAN'T BE ALL ZERO !!

    return:
        - mae, a number shape=()

    """

    # ---- Step 1. Test the shape and weight ---- #
    assert y_true.shape == y_pred.shape, f"`y_true`, `y_pred` should have the SAME shape !"
    assert weight.shape == y_true.shape, f"`weight` should have the SAME shape as y_true&y_pred !"
    assert np.sum(weight) > 0, f"weight can't be all zero !"

    # ---- Step 2. Compute MAE ---- #
    mae = np.sum(weight * np.abs(y_true - y_pred)) / np.sum(weight)  # weighted and mean

    # ---- Step 3. Return MAE score ---- #
    return mae


if __name__ == "__main__":
    y_true = np.array([1, 3, 3])
    y_pred = np.array([1, 2, 3])
    weight = np.array([1, 1, 0])

    r2 = r2_score(y_true=y_true, y_pred=y_pred, weight=weight)
    print("r2 = ", r2)
    corr = corr_score(y_true=y_true, y_pred=y_pred, weight=weight)
    print("corr = ", corr)
    rmse = rmse_score(y_true=y_true, y_pred=y_pred, weight=weight)
    print("rmse = ", rmse)
    mae = mae_score(y_true=y_true, y_pred=y_pred, weight=weight)
    print("mae = ", mae)
