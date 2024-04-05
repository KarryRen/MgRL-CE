# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 09:55
# @Author  : Karry Ren

""" Some util functions. """

import random
import os
import numpy as np
import torch


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
