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
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from utils import fix_random_seed
import elect_config as config
from datasets.elect_dataset import ELECTDataset
from model.MgRL import MgRLNet
from model.loss import MgRL_Loss
from model.metrics import corr_score, rmse_score, mae_score


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
    logging.info(f"**** VALID DATASET & DATALOADER ****")
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

    # ---- Start Train and Valid ---- #
    # init the metric dict of all epochs
    epoch_metric = {
        # train & valid loss
        "train_loss": np.zeros(config.EPOCHS),
        "valid_loss": np.zeros(config.EPOCHS),
        # train & valid CORR
        "train_CORR": np.zeros(config.EPOCHS),
        "valid_CORR": np.zeros(config.EPOCHS),
        # train & valid RMSE
        "train_RMSE": np.zeros(config.EPOCHS),
        "valid_RMSE": np.zeros(config.EPOCHS),
        # train & valid MAE
        "train_MAE": np.zeros(config.EPOCHS),
        "valid_MAE": np.zeros(config.EPOCHS)
    }

    # train model epoch by epoch
    logging.info("***************** BEGIN TRAINING `MgRL` ! *****************")
    # start train and valid during train
    for epoch in tqdm(range(config.EPOCHS)):
        # start timer for one epoch
        t_start = datetime.now()
        # set the array for one epoch to store (all empty)
        train_loss_one_epoch, valid_loss_one_epoch = [], []
        train_dataset_len, valid_dataset_len = len(train_dataset), len(valid_dataset)
        train_preds_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        train_labels_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        train_weights_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        valid_preds_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        valid_labels_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        valid_weights_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        # - train model
        last_step = 0
        model.train()
        for batch_data in tqdm(train_loader):
            # move data to device, and change the dtype from double to float32
            mg_features = batch_data["mg_features"]
            labels = batch_data["label"].to(device=device, dtype=torch.float32)
            weights = batch_data["weight"].to(device=device, dtype=torch.float32)
            # zero_grad, forward, compute loss, backward and optimize
            optimizer.zero_grad()
            outputs = model(mg_features)
            preds, rec_residuals_tuple = outputs["pred"], outputs["rec_residuals"]
            loss = criterion(y_true=labels, y_pred=preds, rec_residuals=rec_residuals_tuple, weight=weights)
            loss.backward()
            optimizer.step()
            # note the loss of training in one iter
            train_loss_one_epoch.append(loss.item())
            # note the result in one iter
            now_step = last_step + preds.shape[0]
            train_preds_one_epoch[last_step:now_step] = preds[:, 0].detach()
            train_labels_one_epoch[last_step:now_step] = labels[:, 0].detach()
            train_weights_one_epoch[last_step:now_step] = weights[:, 0].detach()
            last_step = now_step
        # note the loss and metrics for one epoch of TRAINING
        epoch_metric["train_loss"][epoch] = np.mean(train_loss_one_epoch)
        epoch_metric["train_CORR"][epoch] = corr_score(y_true=train_labels_one_epoch.cpu().numpy(),
                                                       y_pred=train_preds_one_epoch.cpu().numpy(),
                                                       weight=train_weights_one_epoch.cpu().numpy())
        epoch_metric["train_RMSE"][epoch] = rmse_score(y_true=train_labels_one_epoch.cpu().numpy(),
                                                       y_pred=train_preds_one_epoch.cpu().numpy(),
                                                       weight=train_weights_one_epoch.cpu().numpy())
        epoch_metric["train_MAE"][epoch] = mae_score(y_true=train_labels_one_epoch.cpu().numpy(),
                                                     y_pred=train_preds_one_epoch.cpu().numpy(),
                                                     weight=train_weights_one_epoch.cpu().numpy())

        # - valid model
        last_step = 0
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(valid_loader):
                # move data to device, and change the dtype from double to float32
                mg_features = batch_data["mg_features"]
                labels = batch_data["label"].to(device=device, dtype=torch.float32)
                weights = batch_data["weight"].to(device=device, dtype=torch.float32)
                # forward to compute outputs
                outputs = model(mg_features)
                preds, rec_residuals_tuple = outputs["pred"], outputs["rec_residuals"]
                loss = criterion(y_true=labels, y_pred=preds, rec_residuals=rec_residuals_tuple, weight=weights)
                # note the loss of valid in one iter
                valid_loss_one_epoch.append(loss.item())
                # doc the result in one iter, no matter what label_len is, just get the last one
                now_step = last_step + preds.shape[0]
                valid_preds_one_epoch[last_step:now_step] = preds[:, 0].detach()
                valid_labels_one_epoch[last_step:now_step] = labels[:, 0].detach()
                valid_weights_one_epoch[last_step:now_step] = weights[:, 0].detach()
                last_step = now_step
        # note the loss and all metrics for one epoch of VALID
        epoch_metric["valid_loss"][epoch] = np.mean(train_loss_one_epoch)
        epoch_metric["valid_CORR"][epoch] = corr_score(y_true=valid_labels_one_epoch.cpu().numpy(),
                                                       y_pred=valid_preds_one_epoch.cpu().numpy(),
                                                       weight=valid_weights_one_epoch.cpu().numpy())
        epoch_metric["valid_RMSE"][epoch] = rmse_score(y_true=valid_labels_one_epoch.cpu().numpy(),
                                                       y_pred=valid_preds_one_epoch.cpu().numpy(),
                                                       weight=valid_weights_one_epoch.cpu().numpy())
        epoch_metric["valid_MAE"][epoch] = mae_score(y_true=valid_labels_one_epoch.cpu().numpy(),
                                                     y_pred=valid_preds_one_epoch.cpu().numpy(),
                                                     weight=valid_weights_one_epoch.cpu().numpy())

        # save model&model_config and metrics
        torch.save(model, config.MODEL_SAVE_PATH + f"model_pytorch_epoch_{epoch}")
        pd.DataFrame(epoch_metric).to_csv(config.MODEL_SAVE_PATH + "model_metric.csv")

        # write metric log
        dt = datetime.now() - t_start
        logging.info(f"Epoch {epoch + 1}/{config.EPOCHS}, Duration: {dt}, "
                     f"{['%s:%.4f ' % (key, value[epoch]) for key, value in epoch_metric.items()]}")
    # draw figure of train and valid metrics
    plt.figure(figsize=(15, 6))
    plt.subplot(4, 1, 1)
    plt.plot(epoch_metric["train_loss"], label="train loss", color="g")
    plt.plot(epoch_metric["valid_loss"], label="valid loss", color="b")
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(epoch_metric["train_CORR"], label="train CORR", color="g")
    plt.plot(epoch_metric["valid_CORR"], label="valid CORR", color="b")
    plt.legend()
    plt.subplot(4, 2, 1)
    plt.plot(epoch_metric["train_RMSE"], label="train RMSE", color="g")
    plt.plot(epoch_metric["valid_RMSE"], label="valid RMSE", color="b")
    plt.legend()
    plt.subplot(4, 2, 2)
    plt.plot(epoch_metric["train_MAE"], label="train MAE", color="g")
    plt.plot(epoch_metric["valid_MAE"], label="valid MAE", color="b")
    plt.legend()


if __name__ == "__main__":
    # ---- Step 0. Fix the random seed ---- #
    fix_random_seed(seed=config.RANDOM_SEED)

    # ---- Step 1. Train & Valid model ---- #
    train_valid_model()
