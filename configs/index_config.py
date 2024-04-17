# -*- coding: utf-8 -*-
# @Time    : 2024/4/17 12:59
# @Author  : Karry Ren

""" Config file of CSI300 index dataset. """

# ************************************************************************************ #
# ********************************** BASIC SETTINGS ********************************** #
# ************************************************************************************ #
RANDOM_SEED = [0, 42, 3407, 114514][0]  # the random seed
SAVE_PATH = f"exp_index/rs_{RANDOM_SEED}/"  # the save path of CSI300 index experiments
LOG_FILE = SAVE_PATH + "log.log"  # the log file path
MODEL_SAVE_PATH = SAVE_PATH + "trained_models/"  # the saving path of models
IMAGE_SAVE_PATH = SAVE_PATH + "pred_images/"  # the saving path of pred images

# ************************************************************************************ #
# ************************************ FOR DATASET *********************************** #
# ************************************************************************************ #
INDEX_DATASET_PATH = "../Data/CSI300_index_dataset/dataset"
NEED_NORM = True
TIME_STEPS = 3
BATCH_SIZE = 2048

# ************************************************************************************ #
# ******************************* FOR NET CONSTRUCTING ******************************* #
# ************************************************************************************ #
GRANULARITY_DICT = {"g1": 1, "g2": 4, "g3": 16, "g4": 48, "g5": 240}  # the granularity dict
GA_K, INPUT_SIZE = 1, 6  # the alignment granularity K & the input size
ENCODING_INPUT_SIZE, ENCODING_HIDDEN_SIZE = 1 * GA_K, 64  # the input and hidden size
NEGATIVE_SAMPLE_NUM = 5  # the negative sample number (only work when use `MgRL_CE_Net`)
LOSS_REDUCTION, LAMBDA_1, LAMBDA_2, LAMBDA_THETA = "mean", 1.0, 2.0, 0.001  # loss parameter
LR = 0.01  # the learning rate

# ************************************************************************************ #
# ********************************* FOR NET TRAINING ********************************* #
# ************************************************************************************ #
# ---- Train Model ---- #
EPOCHS = 2

# ---- Main metric using to select models ---- #
MAIN_METRIC = "valid_CORR"
