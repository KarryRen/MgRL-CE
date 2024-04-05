# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 09:36
# @Author  : Karry Ren

""" Config file of UCI electricity dataset. """

# ************************************************************************************ #
# ********************************** BASIC SETTINGS ********************************** #
# ************************************************************************************ #
RANDOM_SEED = [0, 42, 3407, 114514][0]  # the random seed
SAVE_PATH = f"exp_elect_rs_{RANDOM_SEED}/"  # the save path of UCI electricity experiments
LOG_FILE = SAVE_PATH + "log.log"  # the log file path
MODEL_SAVE_PATH = SAVE_PATH + "trained_models/"  # the saving path of models

# ************************************************************************************ #
# ************************************ FOR DATASET *********************************** #
# ************************************************************************************ #
UCI_ELECT_DATASET_PATH = ("/Users/karry/KarryRen/Scientific-Projects/2023-SCU-Graduation-Paper"
                          "/Code/Data/UCI_electricity_dataset/dataset")
TIME_STEPS = 2
BATCH_SIZE = 1024
