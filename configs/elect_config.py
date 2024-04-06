# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 09:36
# @Author  : Karry Ren

""" Config file of UCI electricity dataset. """

# ************************************************************************************ #
# ********************************** BASIC SETTINGS ********************************** #
# ************************************************************************************ #
RANDOM_SEED = [0, 42, 3407, 114514][0]  # the random seed
SAVE_PATH = f"exp_elect/rs_{RANDOM_SEED}/"  # the save path of UCI electricity experiments
LOG_FILE = SAVE_PATH + "log.log"  # the log file path
MODEL_SAVE_PATH = SAVE_PATH + "trained_models/"  # the saving path of models

# ************************************************************************************ #
# ************************************ FOR DATASET *********************************** #
# ************************************************************************************ #
UCI_ELECT_DATASET_PATH = "../Data/UCI_electricity_dataset/dataset"
TIME_STEPS = 2
BATCH_SIZE = 2048

# ************************************************************************************ #
# ******************************* FOR NET CONSTRUCTING ******************************* #
# ************************************************************************************ #
ELECT_GRANULARITY_DICT = {"g1": 1, "g2": 2, "g3": 6, "g4": 24, "g5": 96}  # the granularity dict
GA_K, INPUT_SIZE = 1, 1  # the alignment granularity K & the input size
ENCODING_INPUT_SIZE, ENCODING_HIDDEN_SIZE = 1 * GA_K, 64  # the input and hidden size
LOSS_REDUCTION, LAMBDA_1, LAMBDA_THETA = "mean", 1.0, 0.001  # loss parameter
LR = 0.01  # the learning rate

# ************************************************************************************ #
# ********************************* FOR NET TRAINING ********************************* #
# ************************************************************************************ #
# ---- Train Model ---- #
EPOCHS = 1

# ---- Main metric using to select models ---- #
MAIN_METRIC = "valid_CORR"
