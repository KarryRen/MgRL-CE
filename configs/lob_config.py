# -*- coding: utf-8 -*-
# @Time    : 2024/4/17 13:00
# @Author  : Karry Ren

""" Config file of Future LOB dataset. """

# ************************************************************************************ #
# ********************************** BASIC SETTINGS ********************************** #
# ************************************************************************************ #
RANDOM_SEED = [0, 42, 3407, 114514][0]  # the random seed
SAVE_PATH = f"exp_lob/rs_{RANDOM_SEED}/"  # the save path of Future LOB experiments
LOG_FILE = SAVE_PATH + "log.log"  # the log file path
MODEL_SAVE_PATH = SAVE_PATH + "trained_models/"  # the saving path of models
IMAGE_SAVE_PATH = SAVE_PATH + "pred_images/"  # the saving path of pred images

# ************************************************************************************ #
# ************************************ FOR DATASET *********************************** #
# ************************************************************************************ #
LOB_DATASET_PATH = "../Data/Future_LOB_dataset/IF_M0"
TIME_STEPS = 5
BATCH_SIZE = 2048

# ************************************************************************************ #
# ******************************* FOR NET CONSTRUCTING ******************************* #
# ************************************************************************************ #
GRANULARITY_DICT = {"g1": 1, "g2": 2, "g3": 6, "g4": 60, "g5": 120}  # the granularity dict
GA_K, INPUT_SIZE = 1, 1  # the alignment granularity K & the input size
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
