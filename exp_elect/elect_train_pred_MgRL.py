# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 09:33
# @Author  : Karry Ren

""" Training and Prediction code of `MgRLNet` for UCI electricity dataset.

Training, Validation and Prediction be together !
    Here are two functions:
        - `train_valid_model()` => train and valid model, and save trained models of all epochs.
        - `pred_model()` => use the best model to do prediction.

"""

import os
import logging
import torch
import torch.utils.data as data

from utils import fix_random_seed
import elect_config as config
from datasets.elect_dataset import ELECTDataset
from model.MgRL import MgRLNet
from model.loss import MgRL_Loss


def train_valid_model() -> None:
    # ---- Build up the save directory ---- #
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)

    # ---- Construct the train&valid log file (might be same as prediction) ---- #
    logging.basicConfig(filename=config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logging.info(f"***************** RUN TRAIN DEEP-LOB ! *****************")

    # ---- Get the device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** In device {device}   *****************")

    # ---- Make the dataset and dataloader ---- #
    logging.info(f"***************** BEGIN MAKE DATASET & DATALOADER ! *****************")
    logging.info(f"||| time_steps = {config.TIME_STEPS}, batch size = {config.BATCH_SIZE} |||")
    # make the dataset and dataloader of training
    logging.info(f"**** TRAINING DATASET & DATALOADER ****")
    train_dataset = ELECTDataset(root_path=config.UCI_ELECT_DATASET_PATH, data_type="Train", time_steps=config.TIME_STEPS)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    logging.info(f"**** TRAINING DATASET & DATALOADER ****")
    valid_dataset = ELECTDataset(root_path=config.UCI_ELECT_DATASET_PATH, data_type="Valid", time_steps=config.TIME_STEPS)
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    logging.info("***************** DATASET MAKE OVER ! *****************")
    logging.info(f"Train dataset: length = {len(train_dataset)}")
    logging.info(f"Valid dataset: length = {len(valid_dataset)}")

    # ---- Construct the model and transfer device, while making loss and optimizer ---- #
    model = MgRLNet(
        granularity_dict=config.ELECT_GRANULARITY_DICT, ga_K=config.GA_K,
        encoding_input_size=config.ENCODING_INPUT_SIZE, encoding_hidden_size=config.ENCODING_HIDDEN_SIZE,
        device=device
    )  # the model
    criterion = MgRL_Loss(reduction=config.LOSS_REDUCTION, lambda_1=config.LAMBDA_1)  # the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.LAMBDA_THETA)  # the optimizer


if __name__ == "__main__":
    # ---- Step 0. Fix the random seed ---- #
    fix_random_seed(seed=config.RANDOM_SEED)

    # ---- Step 1. Train & Valid model ---- #
    train_valid_model()
