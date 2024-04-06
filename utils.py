# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 09:55
# @Author  : Karry Ren

""" Some util functions. """

import random
import os
import numpy as np
import torch
import pandas as pd


def fix_random_seed(seed: int) -> None:
    """ Fix the random seed to decrease the random of training.
        Ensure the reproducibility of the experiment.

    :param seed: the random seed number to be fixed

    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_best_model(model_save_path: str, metric: str) -> tuple:
    """ Using the metric to select the best model after training and validation.

    :param model_save_path: the path of saving models
    :param metric: the dependent metric to select best model

    return:
        - model: the best model
        - model_path : the path of best model

    """

    # ---- Step 1. Read the metric df and test the `metric` ---- #
    metric_df = pd.read_csv(model_save_path + "model_metric.csv", index_col=0)
    assert metric in metric_df.columns, f"The metric you want use to select best model `{metric}` is not allowed !"

    # ---- Step 2. Get the path of best epoch model ---- #
    best_epoch = metric_df.index[np.argmax(metric_df[metric].values)]
    model_path = model_save_path + f"model_pytorch_epoch_{best_epoch}"

    # ---- Step 3. Load the best model ---- #
    model = torch.load(model_path)
    return model, model_path
