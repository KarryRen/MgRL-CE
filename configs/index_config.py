# -*- coding: utf-8 -*-
# @Time    : 2024/4/17 12:59
# @Author  : Karry Ren

""" Config file of CSI300 index dataset. """

# ************************************************************************************ #
# ********************************** BASIC SETTINGS ********************************** #
# ************************************************************************************ #
RANDOM_SEED = [0, 42, 913, 3407, 114514][1]  # the random seed
SAVE_PATH = f"exp_index/rs_{RANDOM_SEED}/"  # the save path of CSI300 index experiments
LOG_FILE = SAVE_PATH + "log.log"  # the log file path
MODEL_SAVE_PATH = SAVE_PATH + "trained_models/"  # the saving path of models
IMAGE_SAVE_PATH = SAVE_PATH + "pred_images/"  # the saving path of pred images

# ************************************************************************************ #
# ************************************ FOR DATASET *********************************** #
# ************************************************************************************ #
INDEX_DATASET_PATH = "../Data/CSI300_index_dataset/dataset"
NEED_NORM = True
TIME_STEPS = 7
BATCH_SIZE = 256

# ************************************************************************************ #
# ******************************* FOR NET CONSTRUCTING ******************************* #
# ************************************************************************************ #
GRANULARITY_DICT = {"g1": 1, "g2": 4, "g3": 16, "g4": 48, "g5": 240}  # the granularity dict
FEATURE_DIM = 6  # the feature dim
USE_G = "g1"  # select which G
GA_K, INPUT_SIZE = 1, FEATURE_DIM * GRANULARITY_DICT[USE_G]  # the alignment granularity K & the input size
ENCODING_INPUT_SIZE, ENCODING_HIDDEN_SIZE = FEATURE_DIM * GA_K, 64  # the input and hidden size
DROPOUT_RATIO = 0.0  # the dropout ratio
LR = 0.01  # the learning rate
NEGATIVE_SAMPLE_NUM = 5  # the negative sample number (only work when use `MgRL_CE_Net`)
LOSS_REDUCTION, LAMBDA_1, LAMBDA_2, LAMBDA_THETA = "mean", 1.0, 2.0, 0.001  # loss parameter

# ************************************************************************************ #
# ********************************* FOR NET TRAINING ********************************* #
# ************************************************************************************ #
# ---- Train Model ---- #
EPOCHS = 50

# ---- Main metric using to select models ---- #
MAIN_METRIC = "valid_R2"
