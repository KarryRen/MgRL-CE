# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 09:36
# @Author  : Karry Ren

""" Config file of UCI electricity dataset. """

# ************************************************************************************ #
# ********************************** BASIC SETTINGS ********************************** #
# ************************************************************************************ #
RANDOM_SEED = [0, 42, 913][0]  # the random seed
SAVE_PATH = f"exp_elect/rs_{RANDOM_SEED}/"  # the save path of UCI electricity experiments
LOG_FILE = SAVE_PATH + f"log_{RANDOM_SEED}.log"  # the log file path
MODEL_SAVE_PATH = SAVE_PATH + "trained_models/"  # the saving path of models
IMAGE_SAVE_PATH = SAVE_PATH + "pred_images/"  # the saving path of pred images

# ************************************************************************************ #
# ************************************ FOR DATASET *********************************** #
# ************************************************************************************ #
UCI_ELECT_DATASET_PATH = "../Data/UCI_electricity_dataset/dataset"
TIME_STEPS = 7
BATCH_SIZE = 4096

# ************************************************************************************ #
# ******************************* FOR NET CONSTRUCTING ******************************* #
# ************************************************************************************ #
GRANULARITY_DICT = {"g1": 1, "g2": 2, "g3": 6, "g4": 24, "g5": 96}  # the granularity dict
FEATURE_DIM = 1  # the feature dim
USE_G = "g1"  # select which G
GA_K, INPUT_SIZE = 1, FEATURE_DIM * GRANULARITY_DICT[USE_G]  # the alignment granularity K & the input size
ENCODING_INPUT_SIZE, ENCODING_HIDDEN_SIZE = FEATURE_DIM * GA_K, 64  # the input and hidden size
DROPOUT_RATIO = 0.0  # the dropout ratio
NEGATIVE_SAMPLE_NUM = 5  # the negative sample number (only work when use `MgRL_CE_Net`)
LOSS_REDUCTION, LAMBDA_1, LAMBDA_2, LAMBDA_THETA = "mean", 1.0, 2.0, 0.001  # loss parameter
LR = 0.01  # the learning rate

# ************************************************************************************ #
# ********************************* FOR NET TRAINING ********************************* #
# ************************************************************************************ #
# ---- Train Model ---- #
EPOCHS = 30

# ---- Main metric using to select models ---- #
MAIN_METRIC = "valid_CORR"
